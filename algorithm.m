function [F,obj] = algorithm(X, G, V, Z, F, k, lmd1, lmd2)
% N: totall numbers
% k : clusters
num_views = length(X);
U= cell(1,num_views);
tol=1e-5;% convergence
max_iter=100;

alpha = (1/num_views) * ones(1,num_views);
for iter = 1:max_iter
    % disp(['Iter-', num2str(iter)]);

    % ---------- Update Ui ----------%
    for iv = 1:num_views
        %U{iv}=X{iv}*V{iv}'*invVV{iv};

        Xtmp    = gpuArray(X{iv});
        Vtmp    = gpuArray(V{iv});

        invtmp = inv( Vtmp * Vtmp' );
        Utmp   = Xtmp * Vtmp' * invtmp;

        Utmp   = gather(Utmp);
        
        U{iv}  = Utmp;
    end
    
    % ---------- Update Vi ----------%
    V = updateV(X,U,V,Z,lmd1,num_views);
    

    % ---------- Update Zi ----------%
    FF = F*F';
    for iv = 1:num_views
        Znum = size(Z{iv},1);
        Zi = Z{iv};
        GZG = G{iv} * FF * G{iv}';
        
        Vi=V{iv};
        VtV = Vi'*Vi;

        A = 2 * lmd1 * VtV;

        for j = 1:Znum
            B = 2 *  lmd1 * VtV(:,j) + lmd2 * alpha(iv) * GZG(:,j);
            Zi(:,j) = A \ B;
        end
%         Zi = Zi - diag(diag(Zi));
%         for ii = 1:Znum
%             idx = 1:Znum;
%             idx(ii) = [];
%             Zi(ii,idx) = EProjSimplex_new(Zi(ii,idx));
%         end

%          % constraint 
          Zi(Zi<0)=0;
         % normalization
         colsum = sum(Zi,1)+eps;
         colsum_diag = diag(colsum);
         Zi = Zi * colsum_diag^-1;

%           Z{iv} = (Zi+Zi')/2;
%          symmetry ?

           Z{iv} = Zi;
    end
    
    
    % ---------- Update alpha_i ----------%
    % TrSFF = zeros(1,num_views);
    sumT = 0;
    for iv = 1:num_views
        TrSFF(iv) = trace(G{iv}' * Z{iv} * G{iv} * FF);
        sumT = sumT + TrSFF(iv)^2;
    end
    Q = sqrt(sumT);
    for iv = 1:num_views
        alpha(iv) = TrSFF(iv) / Q;
    end
    alpha = alpha ./ (sum(alpha)+eps);
    
    % ---------- Update F ----------%
    M = 0;
    for iv = 1:num_views
        M = M + alpha(iv) * G{iv}' * Z{iv} * G{iv};
    end
    [F,~]  = eig1((M+M')/2, k, 1,0);
    %F = Schmidt_orthogonalization(F); % othrogonalization
 
    %====obj=====
    sumof3 = 0;
    % sumZ = 0;
    for iv = 1:num_views
        NMFterm = norm(X{iv}-U{iv}*V{iv},'fro')^2;
        selfrepres = lmd1 * norm(V{iv}-V{iv} * Z{iv},'fro')^2;
        traceSF = lmd2 * trace(alpha(iv) * G{iv}' * Z{iv} * G{iv} * (F * F'));
        sumof3 =  sumof3 + NMFterm + selfrepres - traceSF;
        % fprintf('View-%d   NMFterm=%g \t selfrepres=%g \t traceSF=%g \n',iv, NMFterm,selfrepres,traceSF);
    end
    obj(iter) = sumof3;
    % fprintf('obj = %g \n',obj(iter));

    if iter>19 && ((abs(obj(iter)-obj(iter-1))/obj(iter-1) < tol) || obj(iter)<=tol)
    %if iter == 20
        %fprintf('Objective value converge to %g at iteration %d before the maxIteration reached \n',obj(iter),iter);
        break;
    end
end
end

% function [Xs, sig] = mysvt(ol, D, c)
% % min_A  ol ||D-A||_F^2 +||A||_*
%     [U, sig, V] = mySVD(D, c);
%     epsi = 1/(2*ol);
% 
%     sig = sig - epsi;
%     diag_sig = sig >0;
% 
%     sig = sig.*diag_sig;
%     Xs = U*sig*V';
% end