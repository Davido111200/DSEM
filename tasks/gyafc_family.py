from tasks.base import BaseProbInference

class FormalityFamilyProbInferenceForStyle(BaseProbInference):
    def __init__(self, prompt_version):
        super().__init__(prompt_version)

        self.can_be_stratified = False
        self.num_base_shot = 1

    def default_prompt_version(self):
        return "sp"

    def dataset_signature(self):
        return {
            "sample": ("gyafc_family", "Family_Relationships", "train"),
            "result": ("gyafc_family", "Family_Relationships", "test")
        }

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            query = (e["informal"], e["formal"])
            data.append({"query": query})
        return data

    def exemplar_seperator(self):
        if self.prompt_version.startswith("sp"):
            return ".  "
        else:
            raise ValueError(f"AGNews: Not supported prompt_version: {self.prompt_version}")

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def paralell_style_promptify(self, query, return_reference = False, Instruction = ''):
        informal, formal = query
        with_sentence_and_paraphrase = Instruction + f'Original: "{informal}"; Paraphrased: "{formal}"'
        with_sentence = Instruction + f'Original: "{informal}"; Paraphrased: "'

        if return_reference:
            return with_sentence, with_sentence_and_paraphrase, formal
        else:
            return with_sentence, with_sentence_and_paraphrase