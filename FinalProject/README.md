# CSCI611-Final-Project---Denoising-Auto-Encoder

An autoencoder, is a special kind of neural network designed to learn important details and representation of images. It works by taking an image as an input and squeezing it down to a compact format referred to as the latent representation or latent space. This latent space preserves the most important features from the image and makes it useful for tasks such as image denoising. The compressed or latent space representation is fed back through the decoder which reconstructs the original image. By learning to ignore irrelevant variations such as noise and blur the model becomes more effective at restoring corrupted images. 


## References

- Hankare, O. (2023). *Autoencoders Explained*. Medium.
  https://ompramod.medium.com/autoencoders-explained-9196c38af6f6
  - Used as reference for the dense autoencoder architecture in `src/auto_encoder.py`

## Contributors

#### Get started by cloning repo
    - accept invite
    - git clone <SSH>
    - cd <project repo>

#### Work should be done from feature branches
    - git checkout -b <feat:branch name>

#### Keep your active branches updated
    - git fetch origin
    - git rebase origin/main

#### When ready to merge your code into main
    - git add <filepath>
    - git commit -m "feat/fix: meaningful commit message"
    - git push origin <feature branch name>

#### Open a pull request
    - go to team repo
    - click on 'pull requests' tab
    - click compare and open pull request
