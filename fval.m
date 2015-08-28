function obj  = fval(gamma,hf,tk,gk,params)
   %Compute the objective function E=||gamma-tk.gk||^2+lambda*||g-t*h||^2 +
    lambda = params.lambda;
    tv_weight = params.tv_weight;
    F = params.F;
    D = params.D;
    H = params.H;
    %Measurement error
    evect1=gamma-tk.*conj(gk);
    obj1 = sum(evect1(:).*conj(evect1(:)));
    %Mismatching error
    evect2 = gk-H*tk;
    obj2 = lambda*sum(evect2(:).*conj(evect2(:)));
    grad_res = D*tk;
    %Total Variation part
    tv = (abs(grad_res(:,:,1)).^2+abs(grad_res(:,:,2)).^2+1e-15).^0.5;
    obj3 = tv_weight*sum(tv(:));
    obj = obj1 + obj2 + obj3;
    