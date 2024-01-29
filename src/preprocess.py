from utils import is_valid_text

def get_weighted_embedding(ft_model, row, word_weight=0.7):
    """
    Returns weighted embedding given row['Word'] and row['Meaning']
    otherwise returns Word embedding of row['Word']

    @param ft_model: fasttext model
    @param row: pandas row
    @param word_weight: how much weight to put on the Word vector of Word label (Word vector)

    @returns weighted embedding
    """
    if is_valid_text(row['Meaning']):
        word_embedding = ft_model.get_word_vector(row['Word'])
        text_embedding = ft_model.get_sentence_vector(row['Meaning'])
        weighted_embedding = word_weight * word_embedding + (1 - word_weight) * text_embedding
    else:
        weighted_embedding = ft_model.get_word_vector(row['Word'])
    return weighted_embedding

def get_combined_sentence_embedding(ft_model, row):
    """
    Returns sentence embedding of the combined string: "row['Word']:row['Meaning']" if Meaning is valid
    otherwise returns Word embedding of row['Word']

    @param ft_model: fasttext model
    @param row: pandas row

    @returns sentence embedding of the string: "row['Word']:row['Meaning']"
    """
    if is_valid_text(row['Meaning']):
        embedding = ft_model.get_sentence_vector(row['Word'] + ':' + row['Meaning'])
    else:
        embedding = ft_model.get_word_vector(row['Word'])
    return embedding


def split_with_without_text(df):
    # returns df with Meaning, df without Meaning
    return df[df['Meaning'].str.len() > 0], df[df['Meaning'].str.len() <= 0]

def preprocess(df, remove_duplicates=True):
    """
    Preprocessing for the English words dataset
    @param df: pandas dataframe
    @param remove_duplicates: whether to remove duplicate rows with the same 'Word' entry

    @returns pandas dataframe
    """
    # fill NA values with empty string
    df = df.fillna('')

    if remove_duplicates:
        df = df.drop_duplicates(subset=['Word'])

    # remove new lines, they cannot be processed by fasttext using sentence vector
    df = df.replace(r'\n',' ', regex=True)

    return df
