#/bin/bash

zip -r submit.zip learned_policies/**/model.pt\
                  ppo.py \
                  learned_policies/NPG/expert_theta.npy \
                  hyperparameters.yaml \
                  models.py \
                  README.md \
                  requirements.txt \
                  test_npg.py \
                  train_npg.py \
                  npg_utils.py \
                  test_ppo.py \
                  tune.py \
                  answers.pdf
                  