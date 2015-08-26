function obj  = fval(gamma,hf,tk,gk,lambda,tv_weight,nrows,ncols)
   %Compute the objective function E=||gamma-tk.gk||^2+lambda*||g-t*h||^2 +
   %tv_weight*tv(t)
    evect=gamma-tk.*conj(gk);
    obj = norm(evect(:),'fro').^2;
    tkf = fft2(tk);
    gkfiltf =hf.*tkf;
    gkfilt = ifft2(gkfiltf);
    evect = gk-gkfilt;
    %Compute the Total Variation part of the signal
    obj = obj + lambda*norm(evect(:),'fro').^2;
end