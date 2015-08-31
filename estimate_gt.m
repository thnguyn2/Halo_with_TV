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
    smart_init_en = params.smart_init_en;
    
    %Get the operators
    F = params.F;    
  
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
    next_a_tk = nlcg(a_gamma,params,cur_a_tk);          
    obj = fval(a_gamma,next_a_tk,params);
    disp(['Current objective: ' num2str(obj)]);
    cur_a_tk = next_a_tk;
    figure(3);
    plot(cur_a_tk(end/2,:));drawnow;
    figure(4);
    subplot(121);imagesc(cur_a_tk);colorbar;title('Reconstructed phase');    
end

