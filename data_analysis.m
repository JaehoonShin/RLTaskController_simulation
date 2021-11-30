if 0
    % load('X:\RPE_pols\history_results\full_data_20210317_natural.mat')
    FILE_SUFFIX = '_20210325_RPE';
    load('X:\RPE_pols\history_results\full_detail_20210325_RPE.mat')
    cols = {'rpe','rpe1','rpe2','ctrl reward','score','p mb','0','10','20','40','visit','applied_reward'};
    feat = zeros(length(cols),82,18000);
    for ii = 1:length(cols)
        feat(ii,:,:) = squeeze(detail(ii,1,:,2001:20000));
    end
    feat_sbj_mean = zeros(length(cols),82,20);
    feat_sbj_std = zeros(length(cols),82,20);
    for ii = 1:82
        for jj = 1:length(cols)
            feat_sbj_mean(jj,ii,:) = mean(reshape(squeeze(feat(jj,ii,:)),[20,900]),2);
            feat_sbj_std(jj,ii,:) = std(reshape(squeeze(feat(jj,ii,:)),[20,900]),0,2);
        end    
    end
end

figure()
for feat_indx = 1:length(cols)
    subplot(3,4,feat_indx)
%     figure()
    hold on
    for ii = 1:82
        errorbar(1:20,squeeze(feat_sbj_mean(feat_indx,ii,:)), squeeze(feat_sbj_std(feat_indx,ii,:))./sqrt(900))
    end
    title(cols{feat_indx})
    hold off
end