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
    tol = params.tol;
    method = params.method;
    smart_init_en = params.smart_init_en;
    
    %Get the operators
    F = params.F;
    D = params.D;
    H = params.H;
    
    a_gamma = angle(gamma);
  
    if (smart_init_en==0)
      tk = ones(size(gamma));%Normal initialization
    else
        init_eps = 1e-8;%Smart initialization regularization factor
        hipf = 1-hf; %This is the fourier transform of delta - hf filter
        ang_gammaf = F*a_gamma;
        a_tkf0=(conj(hipf).*ang_gammaf)./(abs(hipf).^2+init_eps);%Weiner deconvolution
        a_tk = F'*a_tkf0;       
    end
    obj = fval(gamma,a_tk,params); %Compute the current objective
  
    %Next, solve with the non-linear conjugate gradient
    obj_array(end+1)=obj;
    disp(['Iter ' num2str(0) ': current objective: ' num2str(obj)]);
    for iter=1:niter
         tic
         grad_obj = gfval(gamma,curtk,gk,params);
%                     nexttk = curtk - 1e-7*grad_obj;
%                     curtk = nexttk;
%                     curobj_val = fval(gamma,nexttk,gk,params);
%                     disp(['t iter: ' num2str(titer) ', value: ', num2str(curobj_val)]);
                 end
                 tk = nexttk;
   end
%         obj = fval(gamma,tk,gk,params);  
%         obj_array(end+1)=obj;
%         %Draw the cross section of T in figure(2)
%         figure(6);
%         est_acs=abs(tk(round((nrows+1)/2),:));
%         est_cs = angle(tk(round((nrows+1)/2),:));
%   
%         subplot(121);plot(est_acs);title('Amplitude');drawnow;
%         subplot(122);plot(est_cs-est_cs(1));title('Phase');drawnow;
%        
%         te = toc;
%         disp(['Iter ' num2str(iter) ': current objective: ' num2str(obj) '/'...
%             num2str(te),' (s)']);
%         figure(5);
%         plot(log10(obj_array));drawnow;
%         title('Objective function');
%         ylabel('log10(obj)');xlabel('Iteration');
% 
%     end
%     save(strcat('objective_',method,'_','iter_',num2str(iter),'.mat'),'obj_array');
     
    figure(4);
    subplot(121);imagesc(abs(tk));colorbar;title('Amplitude of tk');
    subplot(122);imagesc(angle(tk)-angle(tk(nrows/2,1)));colorbar;title('Phase of tk');drawnow;
end

function y=A_comp(x,hf,lambda,gk,nrows,ncols,F)
    %This function computes the results of (diag(gk.^2)+lambda*H^H*H)*x
    x = reshape(x,[nrows ncols]);
    y = lambda*(H'*(H*x))+conj(gk).*gk.*x;
    y = y(:);
end

