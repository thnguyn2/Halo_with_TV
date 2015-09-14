function [obj,obj1,obj2,obj3]  = fval(gamma,tk,gk,params)
   %Compute the objective function E=||gamma-tk*conj(gk)||^2+lambda*|gk - h*tk|^2 + beta*TV(a_tk)
    lambda = params.lambda;
    beta = params.beta;
    TV = params.TV;
    H = params.H;
    
    %Measurement error
    evect1 = gamma-tk.*conj(gk);
    obj1 = sum(evect1(:).*conj(evect1(:)));
 
    %Convolution error
    evect2 = gk - H*tk;
    obj2 = lambda*sum(evect2(:).*conj(evect2(:)));
    
    %Total Variational prior
    w = TV*tk;
    %Total Variation part computation
    grad_l1_approx = (w.*conj(w)+1e-15).^(0.5); 
    obj3 = beta*sum(grad_l1_approx(:));
    switch (params.method)
        case 'nclg'
            obj = (obj1+obj2+obj3);
        case 'relax'
            obj = obj1 + obj2;
    end
   