# RLTaskController_simulation

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2019-0-01371, Development of brain-inspired AI with human-like intelligence) and Samsung Research Funding Center of Samsung Electronics under Project Number SRFC-TC1603-52.

#How to use

1. Install Matlab >= 2020 version
2. Install Psychotoolbox package
3. Open the task_main_2020.m file
4. Modify following values
  - sess_opt : 'pre' or 'fmri'
  - sess_num : 1 / 1~5(max) for each 'pre'/'fmri' conditions
  - name : any character string
  - image_seed : 1~5
5.Run the task_main_2020.m

# Potential problems
1. The display screen works incorrectly
  - Modify 'whichScreen' in the SIMUL_arbiteration_fmri_rpe.m
