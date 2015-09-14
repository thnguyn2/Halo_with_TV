function [tk] = estimate_gt(gamma,hf,params)
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
    %Note that this algorithm will solve for |gamma - conj(tk)*gk|^2 +
    %lambda*|gk - h*tk|^2+beta*TV(tk)
    smart_init_en = params.smart_init_en;
    
    %Get the operators
    F = params.F;    
    H = params.H;
  
    if (smart_init_en==0)
      tk = ones(size(gamma));%Normal initialization
    else
        init_eps = 1e-2;%Smart initialization regularization factor
        hipf = 1-hf; %This is the fourier transform of delta - hf filter
        a_gamma = angle(gamma);
        a_gammaf = F*a_gamma;
        a_tkf0=(conj(hipf).*a_gammaf)./(abs(hipf).^2+init_eps);%Weiner deconvolution
        a_tk = F'*a_tkf0;
        tk = exp(i*a_tk);
    end
    gk = ones(size(gamma));
    
    [obj,obj1,obj2,obj3] = fval(gamma,tk,gk,params); %Compute the current objective
  
    %Next, solve with the non-linear conjugate gradient
    cjgamma = conj(gamma);
    %% 
    disp(['Iter ' num2str(0) ': current objective: ' num2str(obj) ', measurement term: ' num2str(obj1) ...
       ', convolution: ' num2str(obj2) ', TV term: ' num2str(obj3)]);
    
    for iteridx = 1:100
        %Outer loop to solve for g
        tkf = fft2(tk);
        gk = (tk.*cjgamma+params.lambda*ifft2(tkf.*hf))./(conj(tk).*tk+params.lambda+1e-8);
        gk2 = imfilter(gk,fspecial('gaussian',[150 150],20),'same');
        gk = gk./exp(i*angle(gk2));%Get rid of the low frequency smooth variation in gk
        
        switch (params.method)
            case 'nclg' %Non-linear conjugate gradient solver
                %Solve for tk given gk with conjugate gradient
                tk = nlcg(gamma,tk,gk,params);          
            case 'relax' %This method just ignore the TV term
                beta2 = norm(gk,'fro');
                beta2sqr = beta2^2;
                rhs = beta2sqr*gamma./conj(gk)+params.lambda*(H'*gk);
                rhsf = fft2(rhs);
                tkf = rhsf./(beta2sqr+params.lambda*abs(hf).^2+1e-8); %Added factor for stability
                tk = ifft2(tkf);
                [obj,obj1,obj2]  = fval(gamma,tk,gk,params);
                disp(['Iter ' num2str(iteridx) ': current objective: ' num2str(obj) ', Mismatch: ' num2str(obj1),...
                ', Conv error: ' num2str(obj2)]);
        end
        figure(1)
        subplot(1,3,1);imagesc(angle(tk));colorbar;title(sprintf('Current estimation tk - iter #%d',iteridx));drawnow;
        subplot(1,3,2);imagesc(angle(gk));colorbar;title(sprintf('Current estimation -gk iter #%d',iteridx));drawnow;
        subplot(1,3,3);imagesc(angle(gamma));colorbar;title('Input arg(gamma)');drawnow;
        

        
    end
    
    
    obj = fval(a_gamma,tk,params);
    disp(['Current objective: ' num2str(obj)]);
    tk = next_a_tk;
    figure(3);
    stairs(tk(end/2,:));drawnow;
    figure(4);
    subplot(121);imagesc(tk);colorbar;title('Reconstructed phase');    
end

