# A Divergence-Aware Dual Strategy (DADS) Mitigation for Speech Models
This repo contains the code for "Mitigating Subgroup Disparities in Speech Models: A Divergence-Aware Dual Strategy", submitted at IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING.

In this repository, you will find the code to replicate our experiments.  
We do not include the datasets used in the paper but you can find in the official website and repositories: 
- [FSC](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) 
- [ITALIC](https://huggingface.co/datasets/RiTA-nlp/ITALIC)
- [IEMOCAP](https://sail.usc.edu/iemocap/)
- [LibriSpeech](https://huggingface.co/datasets/librispeech_asr)

## Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## Experimental Settings 

### Datasets 
We evaluate our approach on four publicly available datasets and three different tasks: 
- Intent Classification (IC): Fluent Speech Commands (FSC) for the English language and ITALIC for Italian.
- Emotion Recognition (ER): IEMOCAP.
- Automatic Speech Recognition (ASR): LibriSpeech.

### Metadata
For the above datasets we consider the following metadata: 
1. Demographic metadata describing the speaker (e.g., gender, age, language fluency level)
2. Factors related to speaking and recording conditions (e.g., duration of silences, number of words, speaking rate, and noise level), 
3. If present, task-related metadata, e.g., intents represented as combinations of action, object, and location for FSC, or action and scenario for ITALIC, and emotions for IEMOCAP.  

We discretize continuous metadata using frequency-based discretization into three distinct ranges, labeled as "low", "medium", and "high". 
Hence, continuous values are categorized into discrete bins based on their respective frequencies within the dataset.   
In the experiments, we explore all subgroups with a minimum frequency $s$ of $0.03$.

### Models
We fine-tune the transformer-based wav2vec 2.0 base (ca. 90M parameters) on the FSC and IEMOCAP datasets, the multilingual XLSR (ca. 300M parameters) on ITALIC, and the Whisper base (ca. 74M parameters) on LibriSpeech. The pre-trained checkpoints of these models are obtained from the [Hugging Face hub](https://huggingface.co/models).

### Metrics 
We evaluated the overall model performance using accuracy and macro F1 scored for FSC, ITALIC, and IEMOCAP. and WER and CER for LibriSpeech.
We also assessed the performance at the subgroup level. 
We focused on the most challenging subgroup, i.e., the subgroup that shows the most substantial decrease in performance compared to the overall average, denoted with $\Delta^-_{max}$. 
We also computed the average divergence on the top $n \in [10, 20, 50]$ subgroups with the highest decrease in performance ($\Delta^-_{avg-n}$ ), along with the average absolute divergence across all identified subgroups ($\mid\Delta^-_{avg-all}\mid$). 

## More Details
More information will be added upon acceptance of the paper.

## License
This code is released under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions, please contact [Alkis Koudounas](mailto:alkis.koudounas@polito.it).