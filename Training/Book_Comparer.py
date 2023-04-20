from clean_data.book_cleaner import BookCleaner
import jellyfish as jf
import pylcs
import pandas as pd
import numpy as np
from joblib import load
import pickle
from warnings import filterwarnings
filterwarnings('ignore')

class Book:
    # 1. We have the title, but not the author or publisher or publication date
    # 2. We have the title and author, but not the publisher or publication date
    # 3. We have the title and publisher, but not the author or publication date
    # 4. We have the author and publisher and publication date, but not the title
    def __init__(self, title = None, author = None, publisher = None, publish_year = None, isbn = None):
        self.title = title
        self.author = author
        self.publisher = publisher
        self.publish_year = publish_year
        self.isbn = isbn
        if not any([self.title, self.author, self.publisher, self.publish_year]):
            raise ValueError('At least one book attribute must be provided')
        self.type = self._calc_type()
    
    def get_title(self):
        return self.title
    
    def get_author(self):
        return self.author
    
    def get_publisher(self):
        return self.publisher
    
    def get_publish_year(self):
        return self.publish_year
    
    def get_type(self):
        return self.type

    def get_isbn(self):
        if self.isbn is None:
            return 'N/A'
        return self.isbn
    
    def _calc_type(self):
        if self.title is None:
            return 'Author/Publisher'
        elif self.author is None:
            if self.publisher is None:
                return 'Title'
            else:
                return 'Title/Publisher'
        elif self.publisher is None:
            return 'Title/Author'
        return 'Full'

class BookComparer:
    def __init__(self, path_to_comparison_set, path_to_trained_models, verbose = False, count = 2):
        self.verbose = verbose
        self.count = count
        self.comparison_set = self._build_comparison_set(path_to_comparison_set)
        self.full_data_trained_network = pickle.load(open(path_to_trained_models+'full_data_trained_network.pkl', 'rb'))
        self.title_only_trained_network = pickle.load(open(path_to_trained_models+'title_only_trained_network.pkl', 'rb'))
        self.title_author_trained_network = pickle.load(open(path_to_trained_models+'title_author_trained_network.pkl', 'rb'))
        self.title_publisher_trained_network = pickle.load(open(path_to_trained_models+'title_publisher_trained_network.pkl', 'rb'))
        self.author_publisher_trained_network = pickle.load(open(path_to_trained_models+'author_publisher_trained_network.pkl', 'rb'))

        self.full_data_scalar = load(open(path_to_trained_models+'full_data_scaler.pkl', 'rb'))
        self.title_only_scalar = load(open(path_to_trained_models+'title_only_scaler.pkl', 'rb'))
        self.title_author_scalar = load(open(path_to_trained_models+'title_author_scaler.pkl', 'rb'))
        self.title_publisher_scalar = load(open(path_to_trained_models+'title_publisher_scaler.pkl', 'rb'))
        self.author_publisher_scalar = load(open(path_to_trained_models+'author_publisher_scaler.pkl', 'rb'))
    
    def _build_comparison_set(self, path_to_comparison_set) -> pd.DataFrame:
        df = pd.read_csv(path_to_comparison_set)
        output = df.copy()
        output.columns = ['isbn_b', 'title_b', 'author_b', 'publisher_b', 'publish_year_b']
        return output
    
    def compare_book(self, book : Book) -> pd.DataFrame:
        text_set = self._build_data_set(book)
        isbns = text_set[['isbn_b', 'isbn_a']]
        text_set.drop(['isbn_b', 'isbn_a'], axis=1, inplace=True)

        clean_set = BookCleaner(text_set).get_clean_df()
        n_df = self._generate_numeric_features(clean_set)

        # Filter out books that are too different
        if book.get_type() in ['Full', 'Title', 'Title/Author', 'Title/Publisher']:
            mask = (n_df['removed_all_title_hamming'] <= 28) & (n_df['removed_all_title_levenshtein'] <= 28) & (n_df['removed_all_title_jaro'] >= .4)
            n_df = n_df[mask].reset_index(drop=True)
            text_set = text_set[mask].reset_index(drop=True)
        if book.get_type() in ['Full', 'Title/Author', 'Author/Publisher']:
            mask =  (n_df['cleaned_author_hamming'] <= 11) & (n_df['cleaned_author_jaro'] >= .5) & (n_df['cleaned_author_damerau'] <= 9)
            n_df = n_df[mask].reset_index(drop=True)
            text_set = text_set[mask].reset_index(drop=True)
        if book.get_type() in ['Full', 'Title/Publisher', 'Author/Publisher']:
            mask = (n_df['removed_all_publisher_hamming'] <= 11) & (n_df['removed_all_publisher_levenshtein'] <= 13)
            n_df = n_df[mask].reset_index(drop=True)
            text_set = text_set[mask].reset_index(drop=True)

        if self.verbose: print(f'Reduced set from {len(clean_set)} rows to {len(n_df[mask])}')

        if book.get_type() == 'Full':
            n_df = self.full_data_scalar.transform(n_df)
            return self._return_recommendation(self.full_data_trained_network.predict_proba(n_df), text_set)
        elif book.get_type() == 'Title':
            if self.verbose: print('Title only search')
            
            n_df = n_df[[col for col in list(n_df.columns) if 'publish' not in col and 'author' not in col]].reset_index(drop = True)
            n_df = self.title_only_scalar.transform(n_df)
            return self._return_recommendation(self.title_only_trained_network.predict_proba(n_df), text_set)
        
        elif book.get_type() == 'Title/Publisher':
            if self.verbose: print('Title & Publisher search')

            n_df = n_df[[col for col in list(n_df.columns) if 'author' not in col and 'publish_year' not in col]].reset_index(drop = True)
            n_df = self.title_publisher_scalar.transform(n_df)
            return self._return_recommendation(self.title_publisher_trained_network.predict_proba(n_df), text_set)
        
        elif book.get_type() == 'Title/Author':
            if self.verbose: print('Title & Author search')

            n_df = n_df[[col for col in list(n_df.columns) if 'publish' not in col]].reset_index(drop = True)
            n_df = self.title_author_scalar.transform(n_df)
            return self._return_recommendation(self.title_author_trained_network.predict_proba(n_df), text_set)
        
        elif book.get_type() == 'Author/Publisher':
            if self.verbose: print('Author & Publisher search')

            n_df = n_df[[col for col in list(n_df.columns) if 'title' not in col]].reset_index(drop = True)
            n_df = self.author_publisher_scalar.transform(n_df)
            return self._return_recommendation(self.author_publisher_trained_network.predict_proba(n_df), text_set)
        
        
    def _build_data_set(self, book : Book) -> pd.DataFrame:
        output = self.comparison_set.copy()

        output['isbn_a'] = book.get_isbn()
        output['title_a'] = book.get_title()
        output['author_a'] = book.get_author()
        output['publisher_a'] = book.get_publisher()
        output['publish_year_a'] = book.get_publish_year()

        return output
    
    def _generate_numeric_features(self, df : pd.DataFrame) -> pd.DataFrame:
        numeric_df = pd.DataFrame()

        colHeaders = ['cleaned_author_', 'cleaned_title_', 'cleaned_publisher_', \
                    'removed_all_title_', 'removed_all_publisher_']

        numeric_df['publish_year_delta'] = abs(df['publish_year_b'] - df['publish_year_a'])

        for col in colHeaders:
            col_a = col + 'a'
            col_b = col + 'b'

            print('Generating for col', col, '     ', end='\r')
            numeric_df[col+'levenshtein'] = df.apply(lambda row : jf.levenshtein_distance(row[col_a], row[col_b]), axis = 1)
            numeric_df[col+'damerau'] = df.apply(lambda row : jf.damerau_levenshtein_distance(row[col_a], row[col_b]), axis = 1)
            numeric_df[col+'hamming'] = df.apply(lambda row : jf.hamming_distance(row[col_a], row[col_b]), axis = 1)
            numeric_df[col+'jaro'] = df.apply(lambda row : jf.jaro_similarity(row[col_a], row[col_b]), axis = 1)
            numeric_df[col+'edit_dist'] = df.apply(lambda row : pylcs.edit_distance(row[col_a], row[col_b]), axis = 1)
        print('Done.                                      ', end='\r')
        return numeric_df

    def _return_recommendation(self, probabilities : np.array, text_set : pd.DataFrame) -> pd.DataFrame:
        return_df = text_set.copy()
        return_df['score'] = probabilities[:,1]
        return_df.sort_values(by='score', ascending=False, inplace=True)
        if not self.verbose: return_df.drop('score', axis=1, inplace=True)
        if self.verbose:
            print('Top 2 results:')
            print(return_df[:2])
            return return_df
        return return_df[:self.count].dropna(how='any', axis = 1)

    def _compare_full_book(self, n_df : pd.DataFrame) -> pd.DataFrame:
        out_df = self.full_data_trained_network.predict_proba(n_df)
        return out_df