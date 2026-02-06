"""IOI (Indirect Object Identification) Dataset."""

# Name pairs for IOI task
NAME_PAIRS = [
    ("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"), ("Emma", "James"),
    ("Lisa", "David"), ("Anna", "Michael"), ("Sophie", "Daniel"), ("Rachel", "Chris"),
    ("Laura", "Kevin"), ("Julia", "Peter"), ("Diana", "Steve"), ("Helen", "Mark"),
    ("Grace", "Paul"), ("Claire", "Andrew"), ("Emily", "Ryan"), ("Olivia", "Nathan"),
    ("Mia", "Lucas"), ("Ella", "Henry"), ("Lily", "Jack"), ("Zoe", "Sam"),
    ("Kate", "Ben"), ("Amy", "Luke"), ("Nina", "Max"), ("Eva", "Leo"), ("Iris", "Adam"),
]

# IOI templates
TEMPLATES = [
    "When {name1} and {name2} went to the store, {name2} gave a drink to",
    "When {name1} and {name2} went to the park, {name2} handed a flower to",
    "When {name1} and {name2} went to the office, {name2} sent an email to",
    "When {name1} and {name2} went to the restaurant, {name2} passed the menu to",
    "When {name1} and {name2} went to the library, {name2} gave a book to",
]

def generate_ioi_prompts(n_pairs=25, n_templates=2):
    """Generate IOI prompts with expected answers."""
    prompts = []
    expected = []
    
    for name1, name2 in NAME_PAIRS[:n_pairs]:
        for template in TEMPLATES[:n_templates]:
            prompts.append(template.format(name1=name1, name2=name2))
            expected.append(name1)  # Indirect object is always name1
    
    return prompts, expected

# Pre-generated datasets
PROMPTS_50, EXPECTED_50 = generate_ioi_prompts(25, 2)
PROMPTS_100, EXPECTED_100 = generate_ioi_prompts(25, 4)
