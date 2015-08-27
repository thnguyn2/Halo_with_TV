function obj  = fval(gamma,hf,tk,gk,params)
   %Compute the objective function E=||gamma-tk.gk||^2+lambda*||g-t*h||^2 +
    lambda = params.lambda;
    F = params.F;
    evect1=gamma-tk.*conj(gk);
    obj1 = sum(abs(evect1(:)).^2);
    evect2 = gk-F'*(hf.*(F*tk));
    obj2 = sum(abs(evect2(:)).^2);
    %Compute the Total Variation part of the signal
    obj = obj1 + lambda*obj2;
end