class NaiveBayes:
    def __init__(self, likelihoods, class_probs):
        self.likelihoods = likelihoods
        self.class_probs = class_probs

    def calculate_probability(self, features):
        """Обчислює ймовірність кожного класу для заданих ознак."""
        probs = {}
        for class_name, class_prob in self.class_probs.items():
            prob = class_prob
            for feature, value in features.items():
                prob *= self.likelihoods[feature][value][class_name]
            probs[class_name] = prob
        return probs

    def normalize_probabilities(self, probs):
        """Нормалізує ймовірності."""
        total = sum(probs.values())
        return {class_name: prob/total for class_name, prob in probs.items()}

# Дані
likelihoods = {
    "Outlook": {"Overcast": {"Yes": 4/10, "No": 0/4}},
    "Humidity": {"High": {"Yes": 3/9, "No": 4/5}},
    "Wind": {"Strong": {"Yes": 3/9, "No": 3/5}}
}
class_probs = {"Yes": 10/14, "No": 4/14}

# Умова
features = {"Outlook": "Overcast", "Humidity": "High", "Wind": "Strong"}

# Обчислення
model = NaiveBayes(likelihoods, class_probs)
probs = model.calculate_probability(features)
normalized_probs = model.normalize_probabilities(probs)

# Виведення результату
print("Ймовірності:")
for class_name, prob in normalized_probs.items():
    print(f"  {class_name}: {prob}")