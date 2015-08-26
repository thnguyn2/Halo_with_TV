
function obj = objective_comp(gamma,hf,tk,gk,lambda,tv_weight,nrows,ncols)
    %Compute the objective function E=||gamma-tk.gk||^2+lambda
    evect=gamma-tk.*conj(gk);
    obj = norm(evect(:),'fro').^2;
    tkf = fft2(tk);
    gkfiltf =hf.*tkf;
    gkfilt = ifft2(gkfiltf);
    evect = gk-gkfilt;
    obj = obj + lambda*norm(evect(:),'fro').^2;
    
end