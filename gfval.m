function grad_res = gfval(gamma,hf,tk,gk,params)
%Compute df/dt at tk in order to solve for tk given the current estimation
%of gk and the current solution tk
%The objective function is f = ||gamma(r)-tk*conj(gk)||^2 + lambda*||gk - h
%* tk||^2 +tv_Weight*sum_over_r{[(Dx(tk)(r)^2 + Dy(tk)^2 +1e-15)]^0.5}
%Author: Tan H. Nguyen
    lambda = params.lambda;
    tv_weight = params.tv_weight;
    F = params.F;
    D = params.D;
    %Compute the derivative of the 1st term
    mismatch_grad = 2*(-gamma + tk.*conj(gk)).*conj(gk);
    
    grad_res = mismatch_grad;
    
    