from utils import is_valid_text

def get_weighted_embedding(ft_model, row, word_weight=0.7):
    """
    Returns weighted embedding given row['word'] and row['text']
    otherwise returns word embedding of row['word']

    @param ft_model: fasttext model
    @param row: pandas row
    @param word_weight: how much weight to put on the word vector of word label (word vector)

    @returns weighted embedding
    """
    if is_valid_text(row['text']):
        word_embedding = ft_model.get_word_vector(row['word'])
        text_embedding = ft_model.get_sentence_vector(row['text'])
        weighted_embedding = word_weight * word_embedding + (1 - word_weight) * text_embedding
    else:
        weighted_embedding = ft_model.get_word_vector(row['word'])
    return weighted_embedding

def get_combined_sentence_embedding(ft_model, row):
    """
    Returns sentence embedding of the combined string: "row['word']:row['text']" if text is valid
    otherwise returns word embedding of row['word']

    @param ft_model: fasttext model
    @param row: pandas row

    @returns sentence embedding of the string: "row['word']:row['text']"
    """
    if is_valid_text(row['text']):
        embedding = ft_model.get_sentence_vector(row['word'] + ':' + row['text'])
    else:
        embedding = ft_model.get_word_vector(row['word'])
    return embedding


def split_with_without_text(df):
    # returns df with text, df without text
    return df[df['text'].str.len() > 0], df[df['text'].str.len() == 0]

def preprocess(df, remove_duplicates=True):
    """
    Preprocessing for the English words dataset
    @param df: pandas dataframe
    @param remove_duplicates: whether to remove duplicate rows with the same 'word' entry

    @returns pandas dataframe
    """
    # fill NA values with empty string
    df = df.fillna('')

    if remove_duplicates:
        df = df.drop_duplicates(subset=['word'])

    # remove new lines, they cannot be processed by fasttext using sentence vector
    df = df.replace(r'\n',' ', regex=True)

    return df
