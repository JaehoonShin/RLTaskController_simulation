function SIMUL_cmd_send(simul_set,TASK_TYPE,ii)
    addpath('\\143.248.30.101\sjh\kdj\TerminalControl');
    addpath('\\143.248.30.101\sjh\kdj\TerminalControl');

    id = 'sjh';
    pw = 'kinggodjh';
    control_mode = {'max-spe','min-spe','max-rpe','min-rpe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe'};
%     control_mode = {'min-rpe', 'max-rpe'}; %, 'min-spe', 'max-spe', 'max-rpe-max-spe', 'min-rpe-min-spe', 'max-rpe-min-spe', 'min-rpe-max-spe'}; 
   % control_mode = {'min-rpe'}; 
    %simul_set = 3;     
    % 0 - nomal, single block training
    % 1 - shuffle_simulation
    % 2 - averaging_simulation
    % 3 - 4 block (1 session) training
    % 4 -
    % 5 -
    % 6 - generate multi subjects
%     FILE_SUFFIX = ['_20210520_' num2str(TASK_TYPE), ''];
%     FILE_SUFFIX = ['_20210616_' num2str(TASK_TYPE), ''];
    FILE_SUFFIX = ['_' num2str(TASK_TYPE), '_20_trials'];
    folderpath = '20210827';
    % 20200904 version means 40-or-not based, rpe targeted, flexible fixed, no
    % transitional probability control condition.s
    % TASK_TYPE = 2021;
    done=[4,19];
    if simul_set == 6
        job.path0 = '/home/sjh/RPE_pols';
        job.name = 'simulation_gen_arbs.py';
        job.argu = [' --episodes=100 --trials=20 ', sprintf(' -n %d',ii),' --file-suffix ', ...
            FILE_SUFFIX, sprintf(' --task-type=%d',TASK_TYPE), ' --PMB_CONTROL=0', ' --Reproduce_BHV=0', ' --delta-control=0.75'];
        job.pwd = job.path0;    
        job.nth = ii;
        JobPython(id,job,'Code',1,rem(ii,4)+1);
%             JobPython(id,job,'Code',1);     
        [out] = SubmitJob(id,pw,job);  
    elseif simul_set ~= 3
        for con=[1:length(control_mode)]
            job.path0 = '/home/sjh/RPE_pols';
            if simul_set == 0                    
                job.name = 'main.py';   
                job.argu = [' -d --episodes 10000 --trials 20 --ctrl-mode=',control_mode{con}, sprintf(' -n %d',ii),' --disable-detail-plot --file-suffix ', ...
                    FILE_SUFFIX, sprintf(' --task-type=%d',TASK_TYPE), ' --PMB_CONTROL=0', ' --Reproduce_BHV=0', ' --delta-control=0.75'];%, ' --mode202010'];
                %PMB_CONTROL=0 : no PMB control
                %Reproduce-BHV=1 : Fix dqn and used previously estimated policy
                %to get the behavior data
            elseif simul_set == 1
                job.name = 'shuffle_simulation.py';
                job.argu = ['--ctrl-mode=',control_mode{con}, sprintf(' --policy-sbj=%d',ii), sprintf(' --task-type=%d',TASK_TYPE)...
                    ' --file-suffix=' FILE_SUFFIX '_delta_control_highest'];
            elseif simul_set == 2   
                job.name = 'simulation_averaging.py';
                job.argu = ['--ctrl-mode=',control_mode{con}, sprintf(' --task-type=%d',TASK_TYPE), ' --file-suffix=_', FILE_SUFFIX ,'_whole_averaging'];
            elseif simul_set == 4
                job.name = 'Data_analysis_new_opt.py';
                job.argu = [sprintf('--sbj=%d',ii) ' --mode=' control_mode{con} sprintf(' --task=%d',TASK_TYPE) ' --file-suffix=', FILE_SUFFIX '_delta_control --folderpath=' folderpath];
            elseif simul_set == 5
                job.name = 'shuffle_simulation_new_opt.py';
                job.argu = ['--ctrl-mode=',control_mode{con}, sprintf(' --policy-sbj=%d',ii), sprintf(' --task-type=%d',TASK_TYPE)...
                    ' --file-suffix=' FILE_SUFFIX '_delta_control_highest'];
            end
            job.pwd = job.path0;    
            job.nth = ii;
            JobPython(id,job,'Code',1,rem(ii,4)+1);
%             JobPython(id,job,'Code',1);     
            [out] = SubmitJob(id,pw,job);   
        end
    else
        job.path0 = '/home/sjh/RPE_pols';
        sc_sets = perms([1,2,3,4,5]);
        sc_sets = sc_sets(:,1:4);
        %simul_set = 3
        for sc_indx = 1:length(sc_sets)
            job.name = 'main.py';
            job.argu = [' -d --episodes 1000 --trials 20 --ctrl-mode=',num2str(sc_sets(sc_indx,1)),num2str(sc_sets(sc_indx,2)),num2str(sc_sets(sc_indx,3)),num2str(sc_sets(sc_indx,4)), sprintf(' -n %d',ii),' --disable-detail-plot --file-suffix ', FILE_SUFFIX, sprintf(' --task-type=%d',TASK_TYPE), ' --PMB_CONTROL=0', ' --Session_block=1'];
            job.pwd = job.path0;
            job.nth = ii*1000+sc_indx;
            JobPython(id,job,'Code',1,rem(ii,3)+1);
            [out] = SubmitJob(id,pw,job);
        end
    end
end
