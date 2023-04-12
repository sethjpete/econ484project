import pandas as pd
import re

class BookCleaner:
    def __init__(self, df : pd.DataFrame):
        self.df = df
        self.df_clean = self._clean(self.df)

    def get_clean_df(self):
        return self.df_clean
    
    def _clean(self, df) -> pd.DataFrame:
        # Clean all author data
        def clean_author(author : str):
            author = str(author)
            author = author.lower().strip()
            if ',' in author:
                author = author.split(',')[1].strip()+author.split(',')[0].strip()
            elif ' ' in author:
                author = author.split(' ')[0].strip()+author.split(' ')[1].strip()
            return re.sub(r'[^a-zA-Z0-9]', '', author)

        # Just lowercase and strip
        def clean_title_publisher(title : str):
            title = str(title)
            title = title.lower().strip()
            return re.sub(r'[^a-zA-Z0-9 ]', '', title)

        # Remove words that may be problematic
        def remove_title_words(title : str):
            title = title.lower()
            remove_list = ['a ', 'an ', 'the ', 'and ', 'nor ', 'but ', 'so ', 'or ', 'yet ', 'this ', 'that ', 'these ', \
                        'those ', 'by ', 'is ', 'be ', 'are ', 'am ', 'was ', 'were ', 'have ', 'has ', 'had ', \
                        'his ', 'her ', 'he ', 'she ', 'of ']
            for item in remove_list:
                title = title.replace(item, '')
            return title

        def remove_book_specific_words(title : str):
            title = title.lower()
            remove_list = [' novel', ' movie', ' book', ' memoir', ' edition',  ' publisher', ' books', ' member'\
                        'novel ', 'movie ', 'book ', 'memoir ', 'edition ',  'publisher ', 'books ', 'member '] 
            for item in remove_list:
                title = title.replace(item, '')
            return title

        def strip_spaces(text : str):
            return re.sub(r'[\ ]', '', text)
        df_clean = df.copy()

        df_clean['cleaned_author_a'] = df.author_a.apply(clean_author)
        df_clean['cleaned_author_b'] = df.author_b.apply(clean_author)
        df_clean.drop(['author_a', 'author_b'], axis = 1, inplace = True)

        df_clean['cleaned_title_a'] = df.title_a.apply(clean_title_publisher)
        df_clean['cleaned_title_b'] = df.title_b.apply(clean_title_publisher)
        df_clean['cleaned_publisher_a'] = df.publisher_a.apply(clean_title_publisher)
        df_clean['cleaned_publisher_b'] = df.publisher_b.apply(clean_title_publisher)
        df_clean.drop(['title_a', 'title_b', 'publisher_a', 'publisher_b'], axis = 1, inplace = True)

        df_clean['removed_common_title_a'] = df_clean.cleaned_title_a.apply(remove_title_words)
        df_clean['removed_common_title_b'] = df_clean.cleaned_title_b.apply(remove_title_words)
        df_clean['removed_all_title_a'] = df_clean.removed_common_title_a.apply(remove_book_specific_words).apply(strip_spaces)
        df_clean['removed_all_title_b'] = df_clean.removed_common_title_b.apply(remove_book_specific_words).apply(strip_spaces)
        df_clean['removed_common_title_a'] = df_clean.removed_common_title_a.apply(strip_spaces)
        df_clean['removed_common_title_b'] = df_clean.removed_common_title_b.apply(strip_spaces)

        df_clean['removed_common_publisher_a'] = df_clean.cleaned_publisher_a.apply(remove_title_words)
        df_clean['removed_common_publisher_b'] = df_clean.cleaned_publisher_b.apply(remove_title_words)
        df_clean['removed_all_publisher_a'] = df_clean.removed_common_publisher_a.apply(remove_book_specific_words).apply(strip_spaces)
        df_clean['removed_all_publisher_b'] = df_clean.removed_common_publisher_b.apply(remove_book_specific_words).apply(strip_spaces)
        df_clean['removed_common_publisher_a'] = df_clean.removed_common_publisher_a.apply(strip_spaces)
        df_clean['removed_common_publisher_b'] = df_clean.removed_common_publisher_b.apply(strip_spaces)

        return df_clean

