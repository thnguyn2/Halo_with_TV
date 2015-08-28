function [gk,tk] = estimate_gt(a_gamma,hf,params)
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
    nrows = size(a_gamma,1);
    ncols = size(a_gamma,2);
    
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
    
  
    if (smart_init_en==0)
      tk = ones(size(gamma));%Normal initialization
    else
        init_eps = 1e-8;%Smart initialization regularization factor
        hipf = 1-hf; %This is the fourier transform of delta - hf filter
        a_gammaf = F*a_gamma;
        a_tkf0=(conj(hipf).*a_gammaf)./(abs(hipf).^2+init_eps);%Weiner deconvolution
        a_tk = F'*a_tkf0;       
    end
    obj = fval(a_gamma,a_tk,params); %Compute the current objective
  
    %Next, solve with the non-linear conjugate gradient
    disp(['Iter ' num2str(0) ': current objective: ' num2str(obj)]);
    cur_a_tk = a_tk;
    for iter=1:niter     
          gradf = gfval(a_gamma,cur_a_tk,params);
          
          next_a_tk = cur_a_tk - 1e-3*gradf;
          obj = fval(a_gamma,next_a_tk,params);
          disp(['Current objective: ' num2str(obj)]);
          cur_a_tk = next_a_tk;
          figure(3);
          plot(cur_a_tk(end/2,:));drawnow;
    end     
    figure(4);
    subplot(121);imagesc(abs(tk));colorbar;title('Amplitude of tk');
    subplot(122);imagesc(angle(tk)-angle(tk(nrows/2,1)));colorbar;title('Phase of tk');drawnow;
end

