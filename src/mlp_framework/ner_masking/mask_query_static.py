import re
import pprint
from src.mlp_framework.ner_masking.mask_query_base import MaskQueryBase

class MaskQueryStatic(MaskQueryBase):

    def __init__(self, keyword_dict):
        """
        Args: keyword_dict (Dict[List[str]])
        """
        self.keyword_dict = keyword_dict

    # Build the regex pattern
    def build_regex_pattern(self, keywords):
        escaped_keywords = [re.escape(keyword) for keyword in keywords]
        # Sort by length descending to prioritize longer phrases
        escaped_keywords.sort(key=lambda x: len(x), reverse=True)
        pattern = r'\b(?:' + '|'.join(escaped_keywords) + r')\b'
        return re.compile(pattern, re.IGNORECASE)

    def replace_with_consistent_placeholders(self,text,pattern,category):
        counter = 1
        replacement_dict = {}

        def replacer(match):
            nonlocal counter
            matched_keyword = match.group().lower()
            if matched_keyword not in replacement_dict:
                placeholder = f'{category}_{counter:02d}'
                replacement_dict[placeholder] = matched_keyword
                counter += 1
            else:
                placeholder = replacement_dict[matched_keyword]
            return f'{{{placeholder}}}'

        replaced_text = pattern.sub(replacer, text)
        return replaced_text, replacement_dict

    def mask_query(self,query):
        """
        Args:
            query (str): the question from user

        Returns:
            Dict[str, any]:
            e.g.
            {
                "original_query": "What is the average score from uber users?",
                "masked_query": "What is the average {column_01} from uber users?",
                "extracted_keyword": e.g.
                {
                    "column": {"column_01": "score"},
            }
        """
        masked_query = query
        extracted_keyword = {}
        dict_column = {}
        for category in self.keyword_dict:
            pattern = self.build_regex_pattern(self.keyword_dict[category])
            masked_query, extracted_keyword = self.replace_with_consistent_placeholders(query,pattern,category)
        dict_column= {"column" : extracted_keyword}
        output_dict = {
            "original_query": query,
            "masked_query": masked_query,
            "extracted_keyword": dict_column
        }
        print("==============Static masking==================")
        pprint.pprint(output_dict)
        print("==============================================")
        return output_dict

