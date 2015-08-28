function [grad_res,mismatch_grad,consist_grad,tv_grad] = gfval(gamma,tk,gk,params)
%Compute df/dt at tk in order to solve for tk given the current estimation
%of gk and the current solution tk
%The objective function is f = ||gamma(r)-tk*conj(gk)||^2 + lambda*||gk - h
%* tk||^2 +tv_Weight*sum_over_r{[(Dx(tk)(r)^2 + Dy(tk)^2 +1e-15)]^0.5}
%Author: Tan H. Nguyen
    lambda = params.lambda;
    tv_weight = params.tv_weight;
    D = params.D;
    H = params.H; %Filtering operator
    %Compute the derivative of the 1st term
    mismatch_grad = 2*(tk.*conj(gk)-gamma).*conj(gk); %Gradient of the ||gamma-tk.conj(gk)||^2 term
    consist_grad = 2*lambda*(H'*(H*tk-gk)); %Gradient of the lambda*||gk-H.tk||^2 term
    
    G_val = D*tk;                           %Gradient results
    Gx = G_val(:,:,1);
    Gy = G_val(:,:,2);
    tv_grad = -tv_weight*(Gx+Gy)./(Gx.^2+Gy.^2+1e-15).^(0.5);
   
    %grad_res = mismatch_grad+consist_grad+tv_grad;
    grad_res = tv_grad;
    
    
    