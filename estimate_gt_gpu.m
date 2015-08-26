function [gk,tk] = estimate_gt_gpu(gamma,hf,niter,lambda,tv_weight,tol,method,smart_init_en)
    %This function compute the estimation for gk and tk given gamma on GPU.
    %GPU is used to speed up the calculation of fft2 and ifft2
    
  
    nrows = size(gamma,1);
    ncols = size(gamma,2);
    gk = ones(size(gamma)).*exp(i*angle(gamma)); %Initial estimation- not important
    gk = cast(gk,'single');
    gkd = gpuArray(gk);
   
    
    gamma = cast(gamma,'single');    
    gammad = gpuArray(gamma);
    cjgammad = conj(gammad);
    
    hfd = gpuArray(hf);
    if (smart_init_en==0)
       tk = ones(size(gamma));%Normal initialization
       tk = cast(tk,'single');
       tkd = gpuArray(tk);
    else
        init_eps = 1e-8;%Smart initialization regularization factor
        hipdf = 1-hfd; %This is the fourier transform of delta - hf filter
        ang_gammadf = fft2(angle(gammad));
        ang_tkdf0=(conj(hipdf).*ang_gammadf)./(abs(hipdf).^2+init_eps);%Weiner deconvolution
        ang_tkd0 =ifft2(ang_tkdf0);
        tkd = exp(i*real(ang_tkd0));
    end
    

    %Next, solve with the iterative method
    obj = objective_comp(gammad,hfd,tkd,gkd,lambda,tv_weight,nrows,ncols);
    %obj_array=zeros(niter+1);
    %obj_array(1)=gather(obj);
    disp(['Iter ' num2str(0) ': current objective: ' num2str(obj)]);
    prevobj = 0;
    for iter=1:niter
        tic;
        if (mod(iter,1000)==0)
        %    lambda = lambda*sqrt(2); %Increase the weight as far as we go
        end
        %First, recover g from t
        tkfd = fft2(tkd);
        gkd = (tkd.*cjgammad+lambda*ifft2(tkfd.*hfd))./(conj(tkd).*tkd+lambda);
         
        %Next, recover t from g
       switch method
            case 'relax'
                betasqr = conj(gkd(:))'*gkd(:);
                rhsd = betasqr*gammad./conj(gkd)+lambda*Hhg_comp(hfd,gkd);
                rhsfd = fft2(rhsd);
                tkfd = rhsfd./(betasqr+lambda*abs(hfd).^2);
                tkd = ifft2(tkfd);
            case 'cg'
                 rhsd = gkd.*gammad + lambda*Hhg_comp(hfd,gkd);        
                 tkd = cgs(@(x)A_comp(x,hfd,lambda,gkd,nrows,ncols),rhsd(:),tol,30); %Just need a few step to get to the min.
                 tkd = reshape(tkd,[nrows ncols]);
           case 'nlcg'
               
       end
          
         

        obj = objective_comp(gammad,hfd,tkd,gkd,lambda,nrows,ncols);
        rel_err = abs(obj-prevobj)/prevobj;
        if (rel_err<0.001)
            break;
        else
            prevobj = obj;
        end
%        obj_array(iter+1)=obj;
        
        %Draw the cross section of T in figure(2)
        figure(6);
        est_acs=abs(tkd(round((nrows+1)/2),:));
        est_cs = angle(tkd(round((nrows+1)/2),:));
  
        subplot(131);plot(est_acs,'r');title('Amplitude');drawnow;
        subplot(132);plot(est_cs-est_cs(1),'r');title('Phase');drawnow;
        
        %Draw the error
        ed = gammad - tkd.*conj(gkd);
        ed_cs=real(ed(round((nrows+1)/2),:));
        subplot(133);plot(ed_cs,'r');title('Gamma error');
        
        %figure(5);
        %plot(log10(obj_array));
        te = toc;
        disp(['Iter ' num2str(iter) ': current objective: ' num2str(obj) '/Relative error: ',num2str(rel_err),'/'...
            num2str(te),' (s)']);
%        
     %   figure(4);
     %   subplot(121);imagesc(abs(tkd));colorbar;title('Amplitude of tk');drawnow;
     %   subplot(122);imagesc(angle(tkd)-angle(tkd(nrows/2,1)));colorbar;title('Phase of tk');drawnow;


    end
    figure(4);
    subplot(121);imagesc(abs(tkd));colorbar;title('Amplitude of tk');drawnow;
    subplot(122);imagesc(angle(tkd)-angle(tkd(nrows/2,1)));colorbar;title('Phase of tk');drawnow;
    gk = gather(gkd);
    tk = gather(tkd);



end

function yd=A_comp(xd,hfd,lambda,gkd,nrows,ncols)
    %This function computes the results of (diag(gk.^2)+lambda*H^H*H)*x
    xd = reshape(xd,[nrows ncols]);
    xfd = fft2(xd);
    HhHfd = conj(hfd).*hfd;
    yfd = lambda*HhHfd.*xfd;
    yd = ifft2(yfd);
    yd = yd + xd.*conj(gkd).*gkd; %This one is faster than abs(gk).^2  
    yd = yd(:);
end

function Hhgd=Hhg_comp(hfd,gkd)
    %This function compute the product H^H*gk
    gkfd = fft2(gkd);
    Hhgfd=conj(hfd).*gkfd;
    Hhgd = ifft2(Hhgfd);      
end