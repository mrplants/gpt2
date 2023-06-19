import unittest
from AbstractDataset import AbstractDataset
from transformers import GPT2Tokenizer

class TestAbstractDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = AbstractDataset()
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.expected_first_abstract_encoded = tokenizer.encode("Stereo matching is one of the widely used techniques for inferring depth from stereo images owing to its robustness and speed. It has become one of the major topics of research since it finds its applications in autonomous driving, robotic navigation, 3D reconstruction, and many other fields. Finding pixel correspondences in non-textured, occluded and reflective areas is the major challenge in stereo matching. Recent developments have shown that semantic cues from image segmentation can be used to improve the results of stereo matching. Many deep neural network architectures have been proposed to leverage the advantages of semantic segmentation in stereo matching. This paper aims to give a comparison among the state of art networks both in terms of accuracy and in terms of speed which are of higher importance in real-time applications.")

    def test_first_abstract(self):
        # Fetch the first abstract from the dataset
        first_abstract = self.dataset[0]

        # Check that the first abstract is the same as the expected abstract
        self.assertListEqual(first_abstract.tolist(),
                             self.expected_first_abstract_encoded)

if __name__ == '__main__':
    unittest.main()