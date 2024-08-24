import nltk
from nltk.tokenize import word_tokenize
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure consistent styling
sns.set(style="whitegrid")

# Define the paths to your text and list files
novel_paths = [
    "J. K. Rowling - Harry Potter 1 - Sorcerer's Stone.txt",
    "J. K. Rowling - Harry Potter 2 - The Chamber Of Secrets.txt",
    "J. K. Rowling - Harry Potter 3 - Prisoner of Azkaban.txt",
    "J. K. Rowling - Harry Potter 4 - The Goblet of Fire.txt"
]

novel_titles = [
    "Sorcerer's Stone",
    "Chamber of Secrets",
    "Prisoner of Azkaban",
    "Goblet of Fire"
]

characters_file_path = "characters"  # Ensure correct file extension


def normalize_word(word):
    """Normalize possessive forms and other variations."""
    word = re.sub(r"'s$", '', word)
    return word.lower()


def load_target_words(file_path):
    """Load target words from the file and return a list."""
    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return words


def count_word_frequencies(text, target_words):
    """Count occurrences of target words in the text."""
    words = word_tokenize(text)
    normalized_words = [normalize_word(word) for word in words if word.isalpha()]
    word_counts = Counter(normalized_words)
    filtered_counts = {word: word_counts.get(word, 0) for word in target_words}
    return filtered_counts


def process_novel(file_path, target_words):
    """Process a novel file to count word frequencies."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading the file {file_path}: {e}")
        return {}

    word_counts = count_word_frequencies(text, target_words)
    return word_counts


def calculate_change_in_frequency(df):
    """Calculate the change in frequency for each character across novels."""
    max_min_diff = df.groupby('Character')['Frequency'].agg(lambda x: x.max() - x.min())
    return max_min_diff


def filter_top_characters(df, top_n=10):
    """Filter the top N characters with the greatest change in frequency."""
    max_min_diff = calculate_change_in_frequency(df)
    top_characters = max_min_diff.nlargest(top_n).index.tolist()
    filtered_df = df[df['Character'].isin(top_characters)]
    return filtered_df


def order_characters_by_max_appearance(filtered_df):
    """Order characters by their maximum appearances across the novels."""
    max_appearance = filtered_df.groupby('Character')['Frequency'].max()
    sorted_characters = max_appearance.sort_values(ascending=False).index.tolist()
    ordered_df = filtered_df.set_index('Character').loc[sorted_characters].reset_index()
    return ordered_df


def prepare_plot_data(novel_paths, novel_titles, target_words):
    """Prepare data for plotting."""
    data = []
    for path, title in zip(novel_paths, novel_titles):
        frequencies = process_novel(path, target_words)
        for character in target_words:
            freq = frequencies.get(character, 0)
            # Replace zero frequency with a small value to display on log scale
            freq = freq if freq > 0 else None  # Set to None to exclude zeros
            data.append({
                'Novel': title,
                'Character': character.capitalize(),
                'Frequency': freq
            })
    df = pd.DataFrame(data)
    return df


def plot_character_frequencies(df):
    """Plot character frequencies across novels with a logarithmic y-axis."""
    plt.figure(figsize=(12, 8))

    # Remove entries with None frequencies
    df_clean = df.dropna(subset=['Frequency'])

    # Create lineplot
    sns.lineplot(
        data=df_clean,
        x='Novel',
        y='Frequency',
        hue='Character',
        marker='o',
        palette='tab10',  # Adjust based on the number of filtered characters
        linewidth=2,
        hue_order=df_clean['Character'].unique()  # Order based on maximum appearances
    )

    plt.yscale('log')
    plt.ylabel('Number of Appearances (Log Scale)')
    plt.xlabel('Novel Titles')
    plt.title('Character Appearance Frequency Across Harry Potter Novels')
    plt.xticks(rotation=45)
    plt.legend(title='Characters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('ordered_character_frequency_lineplot_log_scale.png', dpi=300)



if __name__ == "__main__":
    target_words = load_target_words(characters_file_path)
    plot_df = prepare_plot_data(novel_paths, novel_titles, target_words)

    # Filter the top characters with the greatest frequency change
    filtered_df = filter_top_characters(plot_df, top_n=10)

    # Order the filtered characters by their maximum appearance
    ordered_df = order_characters_by_max_appearance(filtered_df)

    # Plot the ordered data
    plot_character_frequencies(ordered_df)
