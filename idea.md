In research.md is a research: Dual Cross-Attention Learning (DCAL) for Fine-Grained Visual Categorization and Object Re-Identification. Give it a look.

### Purpose:
- Adapt the idea of DCAL to a similar task: identical twin faces verification (given two highly similar face image, tell whether they are same person or not). Twin faces look highy similar, except for small details - which can be consider as an fine-grained image analysis task.
- Sadly, the authors of DCAL didn't public their code, so we have to both implement from scratch the DCAL and adapt the DCAL to our task.

### Dataset for training:
- Dataset for training is the ND TWIN 2009 - 2010 dataset. Faces images were pre-processed through: face landmarks detection, face alignment, face detection, cropped out face-only images. Remember to tell me the size to resize to, no matter square ratio or rectangle, a dataset is ready.
- The dataset are already splitted into train and test sets. The file train_dataset_infor.json shows the structure of the train dataset: under a dictionary, with keys are person ids and corresponding value of each key is a list of image paths. The train_twin_pairs.json contains the information on twin pairs under a list of twin person id pairs. The same with test set: test_dataset_infor.json and test_twin_pairs.json. Take a look at them to know the format.
- There are 178 twin pairs, 356 person ids, the number of images per person is minimum of 4, maximum is 68, total image are 6182. Test set has 62 ids - 31 twin pairs, 907 images. Take a look at some of their first lines for more information.

### Training resources:
- Kaggle (Nvidia Tesla P100 - 16GB VRAM).

### Evaluation methods:
- EER, AUC, ROC, Verification Accuracy. Calculate metrics focus on twin only and same person, we do not care non twin.

### Tracking method:
- For Kaggle, API key is in Kaggle kaggle secrets, in WANDB_API_KEY variable. The entity name is hunchoquavodb-hanoi-university-of-science-and-technology.

### More information:
- Model checkpoint should be performed every 5 epoch.
- There should be two option: fresh training or resume training from checkpoint.
- This repo will be push to GitHub, with all .json file containing dataset information. On Kaggle, we clone the repo from Github:

```Kaggle
!git clone "https://github.com/github_name/this_repo.git"
```

and training:

```python
import os
# Change working dir to our repo
os.chdir('/kaggle/working/repo_name')

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
print("WandB configured")

# Run training
!python ...
```

### Question:
- Is the idea of DCAL suitable for my task? (Consider the downstream task characteristic, training data size, etc)?
- Since there is no official implement, many things mentioned in original paper may strange to you. Read the research carefully, list them down, I will find the official implementation on GitHub for you to look at?
- What is the input size images I should resize to?
- If you still have any question, tell me?

### Things to do:
- Write an implementation plan markdown file, for YOU to revisit while implement the idea. This file's purpose is to prevent you from forget what you are doing. It should be clear, fully in detail but short and concise. The content can be the project structure, what you are writing in this files, etc...
- Break down implementation into phases, to do list. We will go step by step.
- In the original paper, they apply attention map over images. This is interesting, it give me an eplainable way to know how the model tell what face part make twin people different. If this is possible, add it in a later phase in implement plan.
- I want a code base that efficient, clear, just write code that needed. The code base should fully utilized both training datset and training resources. Another time, be minimal, just write scripts/code that necessary. I do not need notebooks, utils that never used.