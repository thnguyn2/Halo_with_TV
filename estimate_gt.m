function [gk,tk] = estimate_gt(gamma,hf,params)
    %This function compute the estimation for gk and tk given gamma
    %Author: T. H. Nguyen
    %Date: 08/26/2015
    %Inputs:
    %   gamma: gamma_t,r(r,r,0)
    %   hf: 2D Fourier transform of the spatial correlation function
    %   params: a struct containing all the parameters and operators used
    %   in the solver.
    %Outputs:
    %   gk, tk: results of the solver
    
    obj_array=zeros(0,1);
    nrows = size(gamma,1);
    ncols = size(gamma,2);
    gk = ones(size(gamma)).*exp(i*angle(gamma)); %Initial estimation
    
    %Read data from struct
    niter = params.niter;
    lambda = params.lambda;
    tv_weight = params.tv_weight;
    tol = params.tol;
    method = params.method;
    smart_init_en = params.smart_init_en;
    
    %Get the operator
    F = params.F;
    D = params.D;
    
    
    if (smart_init_en==0)
      tk = ones(size(gamma));%Normal initialization
    else
        init_eps = 1e-8;%Smart initialization regularization factor
        hipf = 1-hf; %This is the fourier transform of delta - hf filter
        ang_gammaf = F*angle(gamma);
        ang_tkf0=(conj(hipf).*ang_gammaf)./(abs(hipf).^2+init_eps);%Weiner deconvolution
        ang_tk0 = F'*ang_tkf0;
        tk = exp(i*real(ang_tk0));
        grad_val = D*ang_tk0;
    end
    
    obj = fval(gamma,hf,tk,gk,params);
    %Next, solve with the iterative method
    obj_array(end+1)=obj;
    cjgamma = conj(gamma);
    disp(['Iter ' num2str(0) ': current objective: ' num2str(obj)]);
    for iter=1:niter
        tic
        %First, recover g from t
        tkf = F*tk;
        gk = (tk.*cjgamma+lambda*(F'*(tkf.*hf)))./(conj(tk).*tk+lambda);
        
        beta = norm(gk,'fro');
        betasqr = beta^2;
        %Next, recover t from g      
        switch method
            case 'relax'
                rhs = betasqr*gamma./conj(gk)+lambda*Hhg_comp(hf,gk,F);
                rhsf = F*rhs;
                tkf = rhsf./(betasqr+lambda*abs(hf).^2);
                tk = F'*tkf;
            case 'cg'
                rhs = gk.*gamma + lambda*Hhg_comp(hf,gk);        
                tk = cgs(@(x)A_comp(x,hf,lambda,gk,nrows,ncols),rhs(:),tol,20,F);
                tk = reshape(tk,[nrows ncols]);
            case 'nlcg' %Non-linear conjugate gradient solver
               

                
        end
        obj = fval(gamma,hf,tk,gk,params);
  
        obj_array(end+1)=obj;
        %Draw the cross section of T in figure(2)
        figure(6);
        est_acs=abs(tk(round((nrows+1)/2),:));
        est_cs = angle(tk(round((nrows+1)/2),:));
  
        subplot(121);plot(est_acs);title('Amplitude');drawnow;
        subplot(122);plot(est_cs-est_cs(1));title('Phase');drawnow;
       
        te = toc;
        disp(['Iter ' num2str(iter) ': current objective: ' num2str(obj) '/'...
            num2str(te),' (s)']);
        figure(5);
        plot(log10(obj_array));drawnow;
        title('Objective function');
        ylabel('log10(obj)');xlabel('Iteration');

    end
    save(strcat('objective_',method,'_','iter_',num2str(iter),'.mat'),'obj_array');
     
    figure(4);
    subplot(121);imagesc(abs(tk));colorbar;title('Amplitude of tk');
    subplot(122);imagesc(angle(tk)-angle(tk(nrows/2,1)));colorbar;title('Phase of tk');drawnpw;
   
   



end

function y=A_comp(x,hf,lambda,gk,nrows,ncols,F)
    %This function computes the results of (diag(gk.^2)+lambda*H^H*H)*x
    x = reshape(x,[nrows ncols]);
    xf = F*x;
    HhHf = conj(hf).*hf;
    yf = lambda*HhHf.*xf;
    y = F'*yf;
    y = y + x.*conj(gk).*gk; %This one is faster than abs(gk).^2  
    y = y(:);
end

function [Hhg,Hhgf]=Hhg_comp(hf,gk,F)
    %This function compute the product H^H*gk
    gkf = F*gk;
    Hhgf = conj(hf).*gkf;
    Hhg = F'*Hhgf;
end