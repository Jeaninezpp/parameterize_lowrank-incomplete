% Pei Zhang 2020-06-25
clear;
clc
warning off;
resultdir = 'results/randmiss0812/';
if (~exist('results/randmiss0812/','file'))
    mkdir('results/randmiss0812/');
    addpath(genpath('results/randmiss0812/'));
end
%% Example Title
datadir = './setting/random remove/';
dataname = {'ORL','orlRnSp','buaaRnSp','caltech7','mfeatRnSp','100Leaves'}; 
% AWGF: 100Leaves buaaRnSp caltech7 mfeatRnSp
datanum = length(dataname);
for datai = 1:6
    datafile = [datadir, cell2mat(dataname(datai))];
    load(datafile); % truth,data,per10-per70,
    fprintf('%s...\n',datafile);
    
    num_clusters = length(unique(truth));
    num_views    = length(data);
    num_sample  = length(truth);

    for per_in = 1:5  % per incomplete ratio 
        in_ratio = per_in*10;
        percent = per{per_in};%percent = cell(1,10);
        disp(['random miss ', num2str(in_ratio)]);
        
        for folds = 1:1
            foldspath = [resultdir, char(dataname(datai)),'/',num2str(in_ratio)];
            if (~exist(foldspath,'file'))
                mkdir(foldspath);
                addpath(genpath([foldspath,'/']));
            end 
            
            index = percent{folds};
            savetxt = [resultdir ,'randmiss_',char(dataname(datai)),'_',num2str(in_ratio),'%','.txt'];
            mes = [char(datetime),'Folds = ', num2str(folds)];
            dlmwrite(savetxt, mes,'-append','delimiter','\t','newline','pc');
            %% get the incomplete Xi, Gi, Vi, Zi
            for iv = 1:num_views
                X_complete = data{iv};% di × n
                X_complete = NormalizeFea(X_complete,1);
                exist_index{iv}  = find(index(:,iv) == 1);
                X_incomplete{iv} = X_complete(:,exist_index{iv});% d_i * n_i
                ni_num(iv) = size(X_incomplete{iv},2);% num of existing sample

                % ===Generate n * n_i incomplete indicator: Gi===
                % If the i-th sample in n is the j-th sample in n_i, the G_ij =1.
                Gtmp = zeros(num_sample,length(exist_index{iv}));% n * n_i
                for gi = 1:length(exist_index{iv})
                    Gtmp(exist_index{iv}(gi),gi)=1;
                end
                G{iv}=Gtmp'; % = W : n_i * n
            end
            clear X X_complete
            X=X_incomplete; % X \in di * ni
            clear X_incomplete

            Zbigsum = zeros(num_sample);
            for iv = 1:num_views
                % ===Initial Vi with kmeans=== k*ni
%                 Vtmp = litekmeans(X{iv}',num_clusters,'MaxIter',100);% Vtmp: ni*1
%                 tmp = zeros(ni_num(iv),num_clusters);
%                 tmp(sub2ind(size(tmp),[1:ni_num(iv)],Vtmp'))=1;% ni*1->ni*k
%                 V{iv} = tmp';
                V{iv} = rand(num_clusters, ni_num(iv));
                
                % ===Initial Zi===
                options = [];
                options.NeighborMode = 'KNN';
                options.k = 3;% num_clusters
                % options.WeightMode = 'Binary';
                Z1 = constructW(X{iv}',options);% input N*d dim X
                Z_ini{iv} = full(Z1);
                clear Z1

                % ===Initial Z*===
                Zbig{iv} = G{iv}'*Z_ini{iv}*G{iv};
                Zbigsum  = Zbigsum + (1/num_views)*Zbig{iv};
            end
 
            [F_ini,~] = eigs(Zbigsum, num_clusters, 'la'); 
                % === finish initial Z*===
            clear  Gtmp Vtmp tmp 


            %% Method
            lambda1 = 10.^[-3:1:3];
            lambda2 = 10.^[-3:1:3];
%             lambda1=1;
%             lambda2=1;

            
            resultsmat = [];
            for i=1:length(lambda1)
                for j = 1:length(lambda2) 
                        disp(['lmd1: ',num2str(lambda1(i)),'    lmd2: ',num2str(lambda2(j))]);
                        for repi = 1:1
                            % disp(['Repeat:',num2str(repi)]);
                            tic;
                            [F, obj]  = algorithm(X, G, V, Z_ini, F_ini ,num_clusters, lambda1(i), lambda2(j));
                            
                            cluster_iter = 1;
                            for ic = 1:cluster_iter
                                Ypred     = kmeans(F, num_clusters, 'replicates',100,'display','off');
                                metric    = clusteringMeasure(truth, Ypred);

                                ACC(ic)    = metric(1);
                                NMI(ic)    = metric(2);
                                Fscore(ic) = metric(3);
                                AR(ic)       = metric(5);
                            end
                            meanACC     = mean(ACC);
                            meanNMI     = mean(NMI);
                            meanFscore  = mean(Fscore);
                            % meanAR        = mean(AR);

                            results = [meanACC, meanNMI, meanFscore];
                            
                            ACC_lmd1_lmd2(i,j) = meanACC;
                            NMI_lmd1_lmd2(i,j) = meanNMI;
                            Fscore_lmd1_lmd2(i,j) = meanFscore;
                            
                            one_repi_time(repi) = toc;
                            disp(['one_repi_time:',num2str(one_repi_time(repi))]);
                        end
                        mean_one_repi_time = mean(one_repi_time);

                        %
                        Final_results = [lambda1(i), lambda2(j), results];

                        % savetxt = [resultdir ,char(dataname),'.txt'];
                        dlmwrite(savetxt, Final_results ,'-append','delimiter','\t','newline','pc');
                        matname = [resultdir, char(dataname(datai)),'/',num2str(in_ratio),'/',num2str(lambda1(i)),'_',num2str(lambda2(j)),'_.mat'];
                        save(matname, 'Final_results','F','Ypred','obj','mean_one_repi_time','ACC','NMI','Fscore');
                        
                        % resultsmat=[resultsmat; Final_results];
                 end
            end
            save([resultdir, char(dataname(datai)), '_',num2str(in_ratio),'%.mat'], 'ACC_lmd1_lmd2','NMI_lmd1_lmd2','Fscore_lmd1_lmd2');
        end % folds end
    end % missing ratio end
end