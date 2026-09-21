## Архитектура финального решения (solution\_v3)

Задача: по 2 452 поисковым запросам выбрать top-50 кандидатов из 189 212 объявлений услуг, метрика — **Recall@50**.

Решение двухстадийное: **многоисточниковый ретривер → CatBoostRanker**.

---

### Стадия 1 — Многоисточниковый пул кандидатов (~600 штук)

Ключевое отличие от v1/v2: вместо одного скора с BM25 — **7 независимых источников**, каждый отбирает свой top-K, потом берётся union.

| Источник | Top-K | Что считает |
|---|---|---|
| `bm25` | 200 | Лексический поиск по title + desc + params (word 1-2gram, k1=1.2, b=0.65) |
| `bm25_geo` | 200 | `bm25 × (1 + 1.25 × affinity)` — гео усиливает релевантные объявления рядом |
| `char` | 100 | Char TF-IDF, n-граммы 3–5 символов — устойчив к опечаткам и морфологии |
| `char_geo` | 100 | `char × (1 + affinity)` |
| `dense` | 200 | Косинусное сходство через `intfloat/multilingual-e5-small` (fine-tuned CachedMNRL) |
| `dense_geo` | 200 | `dense + geo_reward` (аддитивно) |
| `filter_bm25` | 100 | BM25 запроса из поисковых фильтров vs параметры объявлений |
| `lookup` | все | История кликов из train: по запросу, (запрос, категория), (запрос, локация) |

**Union пула** ≈ 600 уникальных кандидатов, полнота ~0.97 Recall.

**Объединение для финального ранжирования пула** — Reciprocal Rank Fusion:
```
RRF(doc) = Σ_source  1 / (60 + rank_source(doc))
```

**BM25** реализован не через rank-bm25, а через разреженные матрицы (`scipy.sparse`): предвычисляется матрица `vocab × n_items`, скоринг запроса — умножение вектора запроса на матрицу, O(vocab).

---

### Гео-фича (принципиальное изменение относительно v1/v2)

В v1/v2 гео-вес 0.50 доминировал над текстом. В v3 — **мягкий бонус/штраф**, не фильтр:

```
affinity = max(same_location, loc_transition_prob, 0.8 × exp(−dist/50km))
geo_boost   = 0.08 × affinity            # max +0.08
geo_penalty = −0.012 × min(log(1 + dist/30km), 6)  # max −0.072
geo_reward  = geo_boost + geo_penalty
```

Для BM25 — мультипликативно: `bm25_geo = bm25 × (1 + 1.25 × affinity)`.  
Для dense — аддитивно: `dense_geo = dense + geo_reward`.

---

### Стадия 2 — CatBoostRanker (PairLogit)

**Не Classifier, а Ranker** — оптимизирует порядок внутри группы запроса.

**Признаки** (21 штука на кандидата):

- Скоры ретривера: `bm25_score`, `bm25_geo_score`, `char_score`, `char_geo_score`, `dense_score`, `dense_geo_score`, `filter_score`, `rrf`
- Гео: `geo_affinity`, `geo_reward`, `same_location`, `loc_prob`, `distance_km`, `log_distance_km`
- Item-признаки: `title_coverage`, `desc_coverage`, `popularity`, `item_known`, `has_rating`, `item_rating`, `log_reviews`
- Нормированные ранги (RRF-нормировка `1/(60+rank)`) для 7 источников: `rrank_bm25`, `rrank_dense`, etc.

**Два прохода обучения с hard negatives:**

1. **Первый проход** — 8000 запросов из train, негативы: top-128 по RRF + 128 случайных из хвоста
2. **Второй проход** — те же запросы, но негативы: **top-128 по прогнозу первой модели** (то, что первая модель ошибочно ставит высоко) + 128 случайных из оставшихся

Параметры: `depth=6`, `lr=0.06`, `iterations=300`, `loss=PairLogit:max_pairs=128`, GPU если доступен.

---

### Валидация

OOD-сплит: 3000 запросов held-out из истории, из них 500 для offline-validation. Метрика — средний Recall@50 на этих запросах.

---

### Эволюция версий

| Версия | Главное изменение |
|---|---|
| v1 | BM25F + гео (вес 0.10) + dense, линейная комбинация скоров |
| v2 | CatBoost Classifier добавлен; гео-вес поднят до 0.50 (слишком сильно) |
| v3 | Многоисточниковый пул (7 каналов) + RRF; гео стал мягким признаком; CatBoostRanker (PairLogit) вместо Classifier; два прохода с hard negatives; char-канал для опечаток |
