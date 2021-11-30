function [varargout]=Permu_data_viewer(task_type,mode)
%% modes
% ''
% 'opt_pol'
% 'opt_bin'
% 'opt_pol_group'
% 'pol_dist'
% '4th'`
% 'ttestgroup'
pol_type = 'new';
switch pol_type
    case 'old'
        folderpath = '/history_results/';
        trials_per_epiosde = 40;
    case 'new'
        folderpath = '/history_results/20210721/';
        trials_per_epiosde = 20;
        file_suffix = ['_' num2str(task_type) '_20_trials_delta_control_highest'];
end

switch task_type
    case 2019
        if strcmp(pol_type,'old'); file_suffix = '_20210616_2019_delta_control'; end;
        goal = {'MIN_RPE','MAX_RPE','MIN_SPE','MAX_SPE','MIN_RPE_MIN_SPE','MAX_RPE_MAX_SPE','MIN_RPE_MAX_SPE','MAX_RPE_MIN_SPE'};
        goal2 = {'min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe'};
    case 2020
         folderpath = '/history_results/20210827/';
        if strcmp(pol_type,'old'); file_suffix = '_20210520_2020_delta_control'; end;
        file_suffix = '_2020_20_trials_delta_control_highest';
        goal = {'MIN_RPE','MAX_RPE','MIN_SPE','MAX_SPE','MIN_RPE_MIN_SPE','MAX_RPE_MAX_SPE','MIN_RPE_MAX_SPE','MAX_RPE_MIN_SPE'}; %goal = {'MIN_RPE','MAX_RPE'};
        goal2 = {'min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe'};% goal2 = {'min-rpe','max-rpe'};
        if strcmp(mode,'pol_dist')
            goal = {'MIN_RPE','MAX_RPE','MIN_SPE','MAX_SPE','MIN_RPE_MIN_SPE','MAX_RPE_MAX_SPE','MIN_RPE_MAX_SPE','MAX_RPE_MIN_SPE'};
            goal2 = {'min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe'};
        end
    case 2021
        if strcmp(pol_type,'old'); file_suffix = '_20210601_2021_delta_control'; end;
        goal = {'MIN_RPE','MAX_RPE','MIN_SPE','MAX_SPE','MIN_RPE_MIN_SPE','MAX_RPE_MAX_SPE','MIN_RPE_MAX_SPE','MAX_RPE_MIN_SPE'};
        goal2 = {'min-rpe','max-rpe','min-spe','max-spe','min-rpe-min-spe','max-rpe-max-spe','min-rpe-max-spe','max-rpe-min-spe'};
    otherwise
        disp('Wrong task type')
end
switch mode
    case ''
        %%%% permutation result viewer
        var_name = {'SPE','RPE','RWD','PMB'};
        for vars = 1:length(var_name)
            figure('name',[var_name{vars}, ' f: ', file_suffix])
            for ii= 1: length(goal)
                load(['\\143.248.30.101\sjh\RPE_pols' folderpath var_name{vars} ' result in the ' goal{ii} 'data' file_suffix '.mat'])
                data = data(:,:,100*trials_per_epiosde+1:end);
                data_tmp = zeros(82,82,100);
                for jj = 1:100
                    data_tmp(:,:,jj) = mean(data(:,:,trials_per_epiosde*jj-trials_per_epiosde+1:trials_per_epiosde*jj),3);
                end
                subplot(4,4,2*ii-1)
                imagesc(mean(data_tmp,3))
                colorbar;
                title([goal{ii} ' mean'])
                xlabel('sbj')
                ylabel('pol')
                subplot(4,4,2*ii)
                imagesc(std(data_tmp,[],3))
                colorbar;
                title([goal{ii} ' std'])
                xlabel('sbj')
                ylabel('pol')
            end
        end
    case 'opt_pol'
        %%%%% opts pol viewer
        var_name = {'SPE','RPE','RWD','PMB'};
        vars = 3;
        threshold = 1;
        for ii= 1:length(goal)
            load(['\\143.248.30.101\sjh\RPE_pols' folderpath var_name{vars} ' result in the ' goal{ii} 'data' file_suffix '.mat'])
            data = data(:,:,100*trials_per_epiosde+1:end);
            data_tmp = zeros(82,82,100);
            for jj = 1:100
                    data_tmp(:,:,jj) = mean(data(:,:,trials_per_epiosde*jj-trials_per_epiosde+1:trials_per_epiosde*jj),3);
            end
            data_goals{ii} = data_tmp;
        end
        for ii= 1:length(goal)
            tmp(ii).tmp1 = zeros(82);
            tmp(ii).tmp2 = zeros(82);
        end
        for k = 1:ceil(82/threshold)
            for ii= 1:length(goal)
                temp_mean = mean(data_goals{ii},3);
                data_maxk(ii).maxk1 = zeros(ceil(82/threshold),82);
                data_maxk(ii).maxk2 = zeros(82,ceil(82/threshold));
                for jj = 1:82
                    if k == ceil(82/threshold)
                        [data_maxk(ii).maxk1(:,jj), indx] = maxk(temp_mean(:,jj),k);
                    else
                        [~, indx] = maxk(temp_mean(:,jj),k);
                    end
                    for k_indx = 1:k
                        tmp(ii).tmp1(indx(k_indx),jj) = tmp(ii).tmp1(indx(k_indx),jj) + 1;
                    end
                end
                for jj = 1:82
                    if k == ceil(82/threshold)
                        [data_maxk(ii).maxk2(jj,:), indx] = maxk(temp_mean(jj,:),k);
                    else
                        [~, indx] = maxk(temp_mean(jj,:),k);
                    end
                    for k_indx = 1:k
                        tmp(ii).tmp2(jj,indx(k_indx)) = tmp(ii).tmp2(jj,indx(k_indx)) + 1;
                    end
                end                
            end
        end
        figure('name',['opt_pol', ' f: ', file_suffix])
        for ii= 1:length(goal)
            subplot(4,4,2*ii-1)
            imagesc(tmp(ii).tmp1)
            colorbar;
            title([goal{ii} ' best policy for sbj'])
            xlabel('sbj')
            ylabel('pol')
            subplot(4,4,2*ii)
            imagesc(tmp(ii).tmp2)
            colorbar;
            title([goal{ii} ' best sbj for policy'])
            xlabel('sbj')
            ylabel('pol')
        end
        figure('name',['opt_pol_rwds', ' f: ', file_suffix])
        for ii= 1:length(goal)
            subplot(4,4,2*ii-1)
            hold on
            bar(mean(data_maxk(ii).maxk1,1))
            errorbar(1:length(mean(data_maxk(ii).maxk1,1)),mean(data_maxk(ii).maxk1,1),std(data_maxk(ii).maxk1,[],1),'.')
            ylim([min(mean(data_maxk(ii).maxk1,1)-std(data_maxk(ii).maxk1,[],1)-1) max(mean(data_maxk(ii).maxk1,1)+std(data_maxk(ii).maxk1,[],1)+1)])
            title([goal{ii} ' opt pol rwds for sbj'])
            xlabel('sbj')
            ylabel('policy averaged ctrl reward')
            hold off
            subplot(4,4,2*ii)
            hold on
            bar(mean(data_maxk(ii).maxk2,2))
            errorbar(1:length(mean(data_maxk(ii).maxk2,2)),mean(data_maxk(ii).maxk2,2),std(data_maxk(ii).maxk2,[],2),'.')
            ylim([min(mean(data_maxk(ii).maxk2,2)-std(data_maxk(ii).maxk2,[],2)-1) max(mean(data_maxk(ii).maxk2,2)+std(data_maxk(ii).maxk2,[],2)+1)])
            title([goal{ii} ' opt sbj rwds for policy'])
            xlabel('pol')
            ylabel('subject averaged ctrl reward')
            hold off
        end
    case 'opt_bin'
        %%%%% opts pol viewer
        var_name = {'SPE','RPE','RWD','PMB'};
        vars = 3;
        threshold = 1;
        num_bins = 10;
        for ii= 1:length(goal)
            load(['\\143.248.30.101\sjh\RPE_pols' folderpath var_name{vars} ' result in the ' goal{ii} 'data' file_suffix '.mat'])
            data = data(:,:,100*trials_per_epiosde+1:end);
            data_tmp = zeros(82,82,100);
            for jj = 1:100
                    data_tmp(:,:,jj) = mean(data(:,:,trials_per_epiosde*jj-trials_per_epiosde+1:trials_per_epiosde*jj),3);
            end
            data_goals{ii} = data_tmp;
        end
        for ii= 1:length(goal)
            tmp(ii).tmp1 = zeros(82);
            tmp(ii).tmp2 = zeros(82);
        end
        for k = 1:ceil(82/threshold)
            for ii= 1:length(goal)
                temp_mean = mean(data_goals{ii},3);
                data_maxk(ii).maxk1 = zeros(ceil(82/threshold),82);
                data_maxk(ii).maxk2 = zeros(82,ceil(82/threshold));
                for jj = 1:82
                    if k == ceil(82/threshold)
                        [data_maxk(ii).maxk1(:,jj), indx] = maxk(temp_mean(:,jj),k);
                    else
                        [~, indx] = maxk(temp_mean(:,jj),k);
                    end
                    if rem(k,ceil(82/threshold/num_bins)) == 0
                        for k_indx = 1:k
                            tmp(ii).tmp1(indx(k_indx),jj) = tmp(ii).tmp1(indx(k_indx),jj) + 1;
                        end
                    end
                end
                for jj = 1:82
                    if k == ceil(82/threshold)
                        [data_maxk(ii).maxk2(jj,:), indx] = maxk(temp_mean(jj,:),k);
                    else
                        [~, indx] = maxk(temp_mean(jj,:),k);
                    end
                    if rem(k,ceil(82/threshold/num_bins)) == 0
                        for k_indx = 1:k
                            tmp(ii).tmp2(jj,indx(k_indx)) = tmp(ii).tmp2(jj,indx(k_indx)) + 1;
                        end
                    end
                end                
            end
        end
        figure('name',['opt_pol', ' f: ', file_suffix])
        for ii= 1:length(goal)
            subplot(4,4,2*ii-1)
            imagesc(tmp(ii).tmp1)
            colorbar;
            title([goal{ii} ' best policy for sbj'])
            xlabel('sbj')
            ylabel('pol')
            subplot(4,4,2*ii)
            imagesc(tmp(ii).tmp2)
            colorbar;
            title([goal{ii} ' best sbj for policy'])
            xlabel('sbj')
            ylabel('pol')
        end
        figure('name',['opt_pol_rwds', ' f: ', file_suffix])
        for ii= 1:length(goal)
            subplot(4,4,2*ii-1)
            hold on
            bar(mean(data_maxk(ii).maxk1,1))
            errorbar(1:length(mean(data_maxk(ii).maxk1,1)),mean(data_maxk(ii).maxk1,1),std(data_maxk(ii).maxk1,[],1),'.')
            ylim([min(mean(data_maxk(ii).maxk1,1)-std(data_maxk(ii).maxk1,[],1)-1) max(mean(data_maxk(ii).maxk1,1)+std(data_maxk(ii).maxk1,[],1)+1)])
            title([goal{ii} ' opt pol rwds for sbj'])
            xlabel('sbj')
            ylabel('policy averaged ctrl reward')
            hold off
            subplot(4,4,2*ii)
            hold on
            bar(mean(data_maxk(ii).maxk2,2))
            errorbar(1:length(mean(data_maxk(ii).maxk2,2)),mean(data_maxk(ii).maxk2,2),std(data_maxk(ii).maxk2,[],2),'.')
            ylim([min(mean(data_maxk(ii).maxk2,2)-std(data_maxk(ii).maxk2,[],2)-1) max(mean(data_maxk(ii).maxk2,2)+std(data_maxk(ii).maxk2,[],2)+1)])
            title([goal{ii} ' opt sbj rwds for policy'])
            xlabel('pol')
            ylabel('subject averaged ctrl reward')
            hold off
        end
    case 'opt_pol_group'
        %%%%% opts pol viewer
        var_name = {'SPE','RPE','RWD','PMB'};
        vars = 3;
        for ii= 1:length(goal)
            load(['\\143.248.30.101\sjh\RPE_pols' folderpath var_name{vars} ' result in the ' goal{ii} 'data' file_suffix '.mat'])
            data = data(:,:,100*trials_per_epiosde+1:end);
            data_tmp = zeros(82,82,100);
            for jj = 1:100
                    data_tmp(:,:,jj) = mean(data(:,:,trials_per_epiosde*jj-trials_per_epiosde+1:trials_per_epiosde*jj),3);
            end
            data_goals{ii} = data_tmp;
        end
        for ii= 1:length(goal)
            tmp(ii).tmp1 = zeros(82);
            tmp(ii).tmp2 = zeros(82);
        end
        for ii= 1:length(goal)
            temp_mean = mean(data_goals{ii},3);
            for jj = 1:82
                indx = find(temp_mean(:,jj) >= temp_mean(jj,jj));
                for k_indx = 1:length(indx)
                    tmp(ii).tmp1(indx(k_indx),jj) = tmp(ii).tmp1(indx(k_indx),jj) + 1;
                end
            end
            for jj = 1:82
                indx = find(temp_mean(jj,:) >= temp_mean(jj,jj));
                for k_indx = 1:length(indx)
                    tmp(ii).tmp2(jj,indx(k_indx)) = tmp(ii).tmp2(jj,indx(k_indx)) + 1;
                end
            end                
        end
        figure('name',['opt_pol', ' f: ', file_suffix])
        for ii= 1:length(goal)
            subplot(4,4,2*ii-1)
            imagesc(tmp(ii).tmp1)
            colorbar;
            title([goal{ii} ' best policy for sbj'])
            xlabel('sbj')
            ylabel('pol')
            subplot(4,4,2*ii)
            imagesc(tmp(ii).tmp2)
            colorbar;
            title([goal{ii} ' best sbj for policy'])
            xlabel('sbj')
            ylabel('pol')
        end
    case 'pol_dist'
        %%%% policy distribution viewer
        figure('name',['policies in' , ' f: ', file_suffix])
        load(['\\143.248.30.101\sjh\RPE_pols' folderpath 'Policy result in the full data' file_suffix '.mat'])
        for ii= 1:length(goal2)            
            subplot(4,2,ii)
            imagesc(squeeze(data(ii,:,:)))
            colorbar
            xlabel('trial')
            ylabel('sbj')
            title(goal2{ii})
        end
        figure('name',['policies angles ' , ' f: ', file_suffix])
        for ii= 1:length(goal2)
            subplot(4,2,ii)
            for jj = 1:82
                for kk = 1:82
                    tmp_sim(jj,kk) = dot(squeeze(data(ii,jj,:)),squeeze(data(ii,kk,:)))/(norm(squeeze(data(ii,jj,:)))*norm(squeeze(data(ii,kk,:))));
                end
            end
            imagesc(tmp_sim)
            colorbar
            xlabel('pol')
            ylabel('pol')
            title(goal2{ii})
        end
    case '4th'
        %% controllability of optimal policies in the different scenarios
        var_name = {'SPE','RPE','RWD','PMB'};
        vars = 3;
        for ii= 1:length(goal)
            load(['\\143.248.30.101\sjh\RPE_pols' folderpath var_name{vars} ' result in the ' goal{ii} 'data' file_suffix '.mat'])
            data = data(:,:,100*trials_per_epiosde+1:end);
            data_tmp = zeros(82,82,100);
            for jj = 1:100
                data_tmp(:,:,jj) = mean(data(:,:,trials_per_epiosde*jj-trials_per_epiosde+1:trials_per_epiosde*jj),3);
            end
            data_goals{ii} = data_tmp;
        end
        for ii= 1:length(goal)
            tmp(ii).tmp1 = zeros(82);
            tmp(ii).tmp2 = zeros(82);
        end
        for ii= 1:length(goal)
            temp_mean = mean(data_goals{ii},3);
            for jj = 1:82
                indx = find(temp_mean(:,jj) >= temp_mean(jj,jj));
                for k_indx = 1:length(indx)
                    tmp(ii).tmp1(indx(k_indx),jj) = tmp(ii).tmp1(indx(k_indx),jj) + 1;
                end
            end
            for jj = 1:82
                indx = find(temp_mean(jj,:) >= temp_mean(jj,jj));
                for k_indx = 1:length(indx)
                    tmp(ii).tmp2(jj,indx(k_indx)) = tmp(ii).tmp2(jj,indx(k_indx)) + 1;
                end
            end                
        end
        for ii = 1:length(goal)
            con_pol(ii).stat = zeros(3,3); % row: rank, mean, std / column : best policy, stable policy, original policy
            con_pol(ii).ttest = zeros(3,3); %{best policy, stable policy, original policy} x {best policy, stable policy, original policy}
            [~,con_pol(ii).stat(1,1)] = max(mean(tmp(ii).tmp1,2));
            load(['\\143.248.30.101\sjh\RPE_pols' folderpath 'Policy result in the ' goal2{ii} 'data' file_suffix '.mat'])
            opt_pol= data(con_pol(ii).stat(1,1),:);
            save(['\\143.248.30.101\sjh\RPE_pols' folderpath 'Best opt pol in the ' goal2{ii} 'data' file_suffix '.mat'],'opt_pol')
            eps_averaged=mean(data_goals{ii},3);
            con_pol(ii).stat(1,2) = mean(eps_averaged(con_pol(ii).stat(1,1),:));
            con_pol(ii).stat(1,3) = std(eps_averaged(con_pol(ii).stat(1,1),:));
            [~,con_pol(ii).stat(2,1)] = min(std(eps_averaged,[],2));
            con_pol(ii).stat(2,2) = mean(eps_averaged(con_pol(ii).stat(2,1),:));
            con_pol(ii).stat(2,3) = std(eps_averaged(con_pol(ii).stat(2,1),:));
            con_pol(ii).stat(3,1) = -1;
            con_pol(ii).stat(3,2) = mean(squeeze(diag(eps_averaged)));
            con_pol(ii).stat(3,3) = std(squeeze(diag(eps_averaged)));
            for t1 = 1:3 % test stat type 1
                for t2 = 1:3 % test stat type 2
                    if t1 ~=3 && t2 ~= 3
                        [~,con_pol(ii).ttest(t1,t2)]= ttest(squeeze(eps_averaged(con_pol(ii).stat(t1,1),:)'),squeeze(eps_averaged(con_pol(ii).stat(t2,1),:)'));
                    elseif t1 ~= 3
                        [~,con_pol(ii).ttest(t1,t2)]= ttest(squeeze(eps_averaged(con_pol(ii).stat(t1,1),:)'),squeeze(diag(eps_averaged)));
                    elseif t2 ~= 3
                        [~,con_pol(ii).ttest(t1,t2)]= ttest(squeeze(diag(eps_averaged)),squeeze(eps_averaged(con_pol(ii).stat(t2,1),:)'));
                    else
                        [~,con_pol(ii).ttest(t1,t2)]= ttest(squeeze(diag(eps_averaged)),squeeze(diag(eps_averaged)));
                    end
                    if isnan(con_pol(ii).ttest(t1,t2))
                        con_pol(ii).ttest(t1,t2) = 1;
                    end
                end
            end
        end
        varargout{1} = con_pol;
        figure('name', ['controllability compare' , ' f: ', file_suffix])
        for ii= 1:length(goal)
            hold on;
            subplot(ceil(length(goal)/2),2,ii)
            bar(1:3,squeeze(con_pol(ii).stat(1:3,2)))
            errorbar(1:3,squeeze(con_pol(ii).stat(1:3,2)),squeeze(con_pol(ii).stat(1:3,3)),'.')
            xlim([0 4])
            ylim([min(squeeze(con_pol(ii).stat(1:3,2))-squeeze(con_pol(ii).stat(1:3,3)))-0.5,max(con_pol(ii).stat(1:3,2)+squeeze(con_pol(ii).stat(1:3,3)))+0.5])
            title([goal{ii}])
            xlabel('best / stable /ori')
            ylabel('controllability')
            hold off;
        end
        pols = zeros(8,20);
        figure('name', ['best policy' , ' f: ', file_suffix])
        for ii = 1:length(goal)
            subplot(ceil(length(goal)/2),2,ii)
            load(['\\143.248.30.101\sjh\RPE_pols' folderpath 'Best opt pol in the ' goal2{ii} 'data' file_suffix '.mat'])
            pols(ii,:) = opt_pol;
            plot(opt_pol);
            title([goal{ii}])
        end
        
        
    case 'ttestgroup'
        var_name = {'SPE','RPE','RWD','PMB'};
        vars = 3;
        threshold = 82;
        p_threshold = 0.05;
        use_threshold = true;
        for ii= 1:length(goal)
            load(['\\143.248.30.101\sjh\RPE_pols' folderpath var_name{vars} ' result in the ' goal{ii} 'data' file_suffix '.mat'])
            data = data(:,:,100*trials_per_epiosde+1:end);
            data_tmp = zeros(82,82,100);
            for jj = 1:100
                data_tmp(:,:,jj) = mean(data(:,:,trials_per_epiosde*jj-trials_per_epiosde+1:trials_per_epiosde*jj),3);
            end
            data_goals{ii} = data_tmp;
        end
        for ii= 1:length(goal)
            tmp(ii).tmp1 = zeros(82);
            tmp(ii).tmp2 = zeros(82);
            tmp(ii).xindx = zeros(82,1);
            tmp(ii).yindx = zeros(82,1);
        end
        for k = 1:ceil(82/threshold)
            for ii= 1:length(goal)
                temp_mean = mean(data_goals{ii},3);
                data_maxk(ii).maxk1 = zeros(ceil(82/threshold),82);
                data_maxk(ii).maxk2 = zeros(82,ceil(82/threshold));
                for jj = 1:82
                    [data_maxk(ii).maxk1(:,jj), indx] = max(squeeze(temp_mean(:,jj)));
                    for kk = 1:82
                        [~,p] = ttest2(squeeze(data_goals{ii}(indx,jj,:)),squeeze(data_goals{ii}(kk,jj,:)),'Vartype','Unequal');
                        if use_threshold
                            if isnan(p)
                                tmp(ii).tmp1(kk,jj) = 1;
                            else
                                tmp(ii).tmp1(kk,jj) = (p > p_threshold);
                            end
                        else
                            if isnan(p)
                                tmp(ii).tmp1(kk,jj) = 1;
                            else
                                tmp(ii).tmp1(kk,jj) = p;
                            end
                        end
                    end
                end
                for jj = 1:82
                    [data_maxk(ii).maxk2(jj,:), indx] = max(squeeze(temp_mean(jj,:)));
                    for kk = 1:82
                        [~,p] = ttest2(squeeze(data_goals{ii}(jj,indx,:)),squeeze(data_goals{ii}(jj,kk,:)),'Vartype','Unequal');
                        if use_threshold
                            if isnan(p)
                                tmp(ii).tmp2(jj,kk) = 1;
                            else
                                tmp(ii).tmp2(jj,kk) = (p > p_threshold);
                            end
                        else
                            if isnan(p)
                                tmp(ii).tmp2(jj,kk) = 1;
                            else
                                tmp(ii).tmp2(jj,kk) = p;
                            end
                        end
                    end
                end
            end
        end
        figure('name',['opt_pol', ' f: ', file_suffix])
        for ii= 1:length(goal)
            subplot(4,4,2*ii-1)
            imagesc(tmp(ii).tmp1)
            colorbar;
            title([goal{ii} ' best policy for sbj'])
            xlabel('sbj')
            ylabel('pol')
            subplot(4,4,2*ii)
            imagesc(tmp(ii).tmp2)
            colorbar;
            title([goal{ii} ' best sbj for policy'])
            xlabel('sbj')
            ylabel('pol')
        end
        figure('name',['sorted opt pol', ' f: ', file_suffix])
        for ii= 1:length(goal)
            subplot(4,4,2*ii-1)
            [~,tmp(ii).xindx] = sort(sum(tmp(ii).tmp1));
            [~,tmp(ii).yindx] = sort(sum(tmp(ii).tmp1,2));
            tmp_plot = tmp(ii).tmp1(:,tmp(ii).xindx);
%             tmp_plot = tmp(ii).tmp1(:,tmp(ii).yindx);
%             tmp_plot = tmp_plot(tmp(ii).yindx,:);
            tmp_plot = tmp_plot(tmp(ii).xindx,:);
            disp(tmp(ii).xindx)
            disp(tmp(ii).yindx')
            imagesc(tmp_plot)
            colorbar;
            title([goal{ii} ' best policy for sbj'])
            xlabel('sbj')
            ylabel('pol')
            subplot(4,4,2*ii)
            tmp_plot = zeros(82,82);
            for tmp_indx = 1:length(tmp(ii).xindx)
                tmp_plot(tmp(ii).yindx(tmp_indx),tmp(ii).xindx(tmp_indx)) = 1;
            end
            imagesc(tmp_plot)
            colorbar;
            title([goal{ii}])
            xlabel('sbj')
            ylabel('pol')
        end
        figure('name',['more sorted opt pol', ' f: ', file_suffix])
        for ii= 1:length(goal)
            subplot(4,4,2*ii-1)
            [~,tmp(ii).xindx] = sort(sum(tmp(ii).tmp1));
            [~,tmp(ii).yindx] = sort(sum(tmp(ii).tmp1,2));
            tmp_plot = tmp(ii).tmp1(:,tmp(ii).xindx);
%             tmp_plot = tmp(ii).tmp1(:,tmp(ii).yindx);
            tmp_plot = tmp_plot(tmp(ii).yindx,:);
%             tmp_plot = tmp_plot(tmp(ii).xindx,:);
            disp(tmp(ii).xindx)
            disp(tmp(ii).yindx')
            imagesc(tmp_plot)
            colorbar;
            title([goal{ii} ' best policy for sbj'])
            xlabel('sbj')
            ylabel('pol')
            subplot(4,4,2*ii)
            tmp_plot = zeros(82,82);
            for tmp_indx = 1:length(tmp(ii).xindx)
                tmp_plot(tmp(ii).yindx(tmp_indx),tmp(ii).xindx(tmp_indx)) = 1;
            end
            imagesc(tmp_plot)
            colorbar;
            title([goal{ii}])
            xlabel('sbj')
            ylabel('pol')
        end
        figure('name',['num of accepted policies : ', ' f: ', file_suffix])
        for ii = 1:8
            subplot(4,4,2*ii-1)
            hist(sum(tmp(ii).tmp1))
            title([goal{ii} ' num of acceptable policy for sbj'])
            subplot(4,4,2*ii)
            hist(sum(tmp(ii).tmp1,2))
            title([goal{ii} ' num of accepted sbj as policy'])
        end
        varargout{1} = tmp;
    case 'rwd_plot'
        %%%%% opts pol viewer
        var_name = {'SPE','RPE','RWD','PMB'};
        vars = 3;
        threshold = 1;
        num_bins = 10;
        view_x = 7;
        view_y = 6;
        record_mode = true;
        if ~record_mode
            view_x = 2;
            view_y = 2;
            figure()
        end

        for ii= 1:length(goal)
            load(['\\143.248.30.101\sjh\RPE_pols' folderpath var_name{vars} ' result in the ' goal{ii} 'data' file_suffix '.mat'])
            data = data(:,:,100*trials_per_epiosde+1:end);
            data_tmp = zeros(82,82,100);
            for jj = 1:100
                data_tmp(:,:,jj) = mean(data(:,:,trials_per_epiosde*jj-trials_per_epiosde+1:trials_per_epiosde*jj),3);
            end
            data_goals{ii} = data_tmp;
            disp(num2str(ii))
        end
        for ii= 1:length(goal)
            tmp(ii).tmp1 = zeros(82);
            tmp(ii).tmp2 = zeros(82);
        end
        for k = 1:ceil(82/threshold)
            for ii= 1:length(goal)
                temp_mean = mean(data_goals{ii},3);
                data_maxk(ii).maxk1 = zeros(ceil(82/threshold),82);
                data_maxk(ii).maxk2 = zeros(82,ceil(82/threshold));
                for jj = 1:82
                    if k == ceil(82/threshold)
                        [data_maxk(ii).maxk1(:,jj), indx] = maxk(temp_mean(:,jj),k);
                    else
                        [~, indx] = maxk(temp_mean(:,jj),k);
                    end
                    if rem(k,ceil(82/threshold/num_bins)) == 0
                        for k_indx = 1:k
                            tmp(ii).tmp1(indx(k_indx),jj) = tmp(ii).tmp1(indx(k_indx),jj) + 1;
                        end
                    end
                end
                for jj = 1:82
                    if k == ceil(82/threshold)
                        [data_maxk(ii).maxk2(jj,:), indx] = maxk(temp_mean(jj,:),k);
                    else
                        [~, indx] = maxk(temp_mean(jj,:),k);
                    end
                    if rem(k,ceil(82/threshold/num_bins)) == 0
                        for k_indx = 1:k
                            tmp(ii).tmp2(jj,indx(k_indx)) = tmp(ii).tmp2(jj,indx(k_indx)) + 1;
                        end
                    end
                end                
            end
        end
        for ii = 1:length(goal)
            for sbj= 1:82
                if rem(sbj,view_x*view_y) == 1
                    if record_mode 
                        figure('name',['opt_pol_rwds', ' f: ', file_suffix, ' sbj : ', num2str(sbj) '~', num2str(view_x*view_y+sbj-1)])
                    else
                        pause(0.5)
                        clf
                    end
                end
                if rem(sbj,view_x*view_y) == 0
                    subplot(view_x,view_y,view_x*view_y)
                else
                    subplot(view_x,view_y,rem(sbj,view_x*view_y))
                end
                hold on
                [~,temp_order] = sort(data_maxk(ii).maxk1(sbj,:));
                bar(data_maxk(ii).maxk1(sbj,temp_order))
                errorbar(1:length(temp_order),data_maxk(ii).maxk1(sbj,temp_order),std(data_goals{ii}(sbj,temp_order,:),[],3),'.')
                xticks(1:length(temp_order))
                xticklabels(string(temp_order))
                ylim([min(data_maxk(ii).maxk1(sbj,:)-std(data_goals{ii}(sbj,:,:),[],3)-1) max(data_maxk(ii).maxk1(sbj,:)+std(data_goals{ii}(sbj,:,:),[],3)+1)])
                title([goal{ii} ' opt pol '  num2str(sbj) ' rwds for sbj'])
                xlabel('sbj')
                ylabel('policy averaged ctrl reward')
                hold off
            end
            for sbj = 1:82
                if rem(sbj,view_x*view_y) == 1
                    if record_mode
                        figure('name',['opt_pol_rwds', ' f: ', file_suffix, ' sbj : ', num2str(sbj) '~', num2str(view_x*view_y+sbj-1)])
                    else
                        pause(0.5)
                        clf
                    end
                end
                if rem(sbj,view_x*view_y) == 0
                    subplot(view_x,view_y,view_x*view_y)
                else
                    subplot(view_x,view_y,rem(sbj,view_x*view_y))
                end
                hold on
                [~,temp_order] = sort(data_maxk(ii).maxk2(:,sbj));
                bar(data_maxk(ii).maxk2(temp_order,sbj))
                errorbar(1:length(temp_order),data_maxk(ii).maxk2(temp_order,sbj),std(data_goals{ii}(temp_order,sbj,:),[],3),'.')
                xticks(1:length(temp_order))
                xticklabels(string(temp_order))
                ylim([min(data_maxk(ii).maxk2(:,sbj)-std(data_goals{ii}(:,sbj,:),[],3)-1) max(data_maxk(ii).maxk2(:,sbj)+std(data_goals{ii}(:,sbj,:),[],3)+1)])
                title([goal{ii} ' opt sbj '  num2str(sbj) ' rwds for pol'])
                xlabel('pol')
                ylabel('subject averaged ctrl reward')
                hold off
            end
        end
    otherwise
        disp('wrong mode')
end
