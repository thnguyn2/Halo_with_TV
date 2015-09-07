function [obj,obj1,obj2]  = fval_linear(a_gamma,a_tk,params)
   %Compute the objective function E=||arg(gamma)-(delta-h)*arg(a_tk)||^2+lambda*TV(a_tk)
    lambda = params.lambda;
    TV = params.TV;
    H = params.H;
    %Measurement error
    evect1 = a_gamma-H*a_tk;
    obj1 = sum(evect1(:).*conj(evect1(:)));
    w = TV*a_tk;
    %Total Variation part computation
    grad_l1_approx = (w.*conj(w)+1e-15).^(0.5); 
    obj2 = lambda*sum(grad_l1_approx(:));
    obj = (obj1+obj2);
   