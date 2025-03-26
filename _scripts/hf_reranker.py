#%% packages
from sentence_transformers import CrossEncoder

#%% Download from the 🤗 Hub
model = CrossEncoder("tomaarsen/reranker-ModernBERT-base-gooaq-lambda")
# Get scores for pairs of texts
pairs = [
    ['How many calories in an egg', 'There are on average between 55 and 80 calories in an egg depending on its size.'],
    ['How many calories in an egg', 'Egg whites are very low in calories, have no fat, no cholesterol, and are loaded with protein.'],
    ['How many calories in an egg', 'Most of the calories in an egg come from the yellow yolk in the center.'],
]
scores = model.predict(pairs)
print(scores.shape)
# (3,)

#%% Or rank different texts based on similarity to a single text
ranks = model.rank(
    # 'How many calories in an egg',
    'Where is it hot?',
    [
        'There are on average between 55 and 80 calories in an egg depending on its size.',
        'Egg whites are very low in calories, have no fat, no cholesterol, and are loaded with protein.',
        'Most of the calories in an egg come from the yellow yolk in the center.',
        'It is warm in the sun.',
    ]
)
ranks
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]

# %%
