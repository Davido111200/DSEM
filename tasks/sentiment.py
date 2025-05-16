from tasks.base import BaseProbInference

class SentimentProbInferenceForStyle(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        return "sp"

    def dataset_signature(self):
        return {
            "sample": ("sentiment", None, "train"),
            "result": ("sentiment", None, "test")
        }

    def dataset_preprocess(self, raw_data):
        pass

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def paralell_style_promptify(self, query, return_reference = False, Instruction = ''):
        # toxic, neutral = query

        # with_sentence_and_paraphrase = Instruction + f'Original: "{toxic}"; Paraphrased: "{neutral}"'
        # with_sentence = Instruction + f'Original: "{toxic}"; Paraphrased: "'

        # if return_reference:
        #     return with_sentence, with_sentence_and_paraphrase, neutral
        # else:
        #     return with_sentence, with_sentence_and_paraphrase
        pass