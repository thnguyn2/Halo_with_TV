
    clc;
    close all;
    clear all;
    datasource = 2;
    if (datasource==1) %Generate simulation data
        Nx = 256;
        nrows = Nx;
        ncols = Nx;  
        phi = pi/2;
        phase = zeros(nrows,ncols);
        phase(round(nrows*0.35:nrows*0.65),round(ncols*0.35:ncols*0.65))=phi;
        phif = fft2(phase);
        nrows = size(phase,1);
        ncols = size(phase,2);
        x_arr = 1:ncols;
        y_arr = 1:nrows;
        [xx,yy]=meshgrid(x_arr,y_arr);
        slope = 0;
        phi_lin =exp(i*xx*slope);
        a_gammaf = phif.*(1-hf);
        a_gamma = ifft2(a_gammaf);
        a_gamma = a_gamma + 0.01*randn(size(a_gamma));
        figure(1);colormap jet;
        subplot(211);imagesc(phase);axis off;colorbar;title('arg(T)');
        subplot(212);imagesc(a_gamma);axis off;colorbar;title('arg(gamma)');drawnow;
        %Display the cross-section of the image
        mcs=a_gamma(round((nrows+1)/2),:);%Get cross-section of the measurement
        cs = phase(round((nrows+1)/2),:);       
   
        figure(2);
        plot(mcs);hold on;plot(cs,'r');hold on;title('Cross sections for phase');
        legend('Gamma(r)','T(r)');drawnow;       
    else   %Go with the real data
       filename = '130815_C2C12_bgd_org_phase_t';
       datafolder = 'E:\Data for Halo removal from EPFL\Setup 1 (Zeiss Axiovert) 60x\';
       
       a_gamma = cast(imread(strcat(datafolder,filename,'.tif')),'single');
       figure(1);
       imagesc(a_gamma);colormap gray;colorbar;       
       Nx = mean(size(a_gamma,1),size(a_gamma,2));
       a_gamma = a_gamma(1:Nx,1:Nx);
       nrows = Nx;
       ncols = Nx;
       
       %Generate data for the non-linear solver
       h_denoise = fspecial('gaussian',[9 9],0.25);
       a_gamma_denoised = imfilter(a_gamma,h_denoise,'same');
       gamma = exp(i*(a_gamma_denoised));
       
       
    end
    
    %Step 2: create the correlation kernel
    htype = 'gaussian';
    
     switch (htype)
        case {'gaussian'} %A gaussian filter
                    bw = 75; %Bandwidth parameter
                    h=fspecial('gaussian',[round(4*bw)+1 round(4*bw)+1],bw); %Transfer function of the low-pass filter...
                    %Fourier transform the filter
                    h1 = zeros(nrows,ncols);
                    h1(1:size(h,1),1:size(h,2))=h;
                    kernel_size=size(h,1);
                    h1 = circshift(h1,[-round((kernel_size-1)/2) -round((kernel_size-1)/2)]);
                    hf = fft2(h1);
 
        case {'lp'} %Bandpass filter
                    lp_bw = 20;%Bandwidth of the low pass filter in the frequency domain. The smaller it is, the more coherent the field will be 
                    hf = zeros(nrows,ncols);
                    [x,y]=meshgrid(linspace(-ncols/2,ncols/2-1,ncols),linspace(-nrows/2,nrows/2-1,nrows));
                    mask = sqrt(x.^2+y.^2)<lp_bw; %
                    hf(mask)=1;
                    hf = fftshift(hf);
         case {'measured'} %Measured kernel
                    load(strcat('Pillars_data/',filename,'_psf.mat'),'h_mask');
                    h = h_mask;
                    h1 = zeros(nrows,ncols);
                    h1(1:size(h,1),1:size(h,2))=h;
                    kernel_size=size(h,1);
                    h1 = circshift(h1,[-round((kernel_size-1)/2) -round((kernel_size-1)/2)]);
                    hf = fft2(h1);
                    %Normalize the kernel
                    hf = hf/max(abs(hf(:)));

             
    end

    
    
    if (gpuDeviceCount()==0)
        gpu_compute_en = 1; %if there is no gpu, compute the result on cpu instead
    end
    gpu_compute_en = 0;
    
    method = 'relax'; %Choose between the two: 'relax','cg','nlcg'
    
    %Parameter definitions
    params.niter =10; %Number of iterations needed
    params.lambda = 10;
    params.beta = 1;
    params.tol = 1e-5; %Tolerance for the solver to stop
    params.method = 'relax';%Choose between 'relax'/'cg'/'nlcf'
    params.smart_init_en = 1;
    %Operator definitions
    params.F = FFT2(Nx); %Fourier transform operator
    params.TV = TV(); %Total variational prior
    params.H = H(Nx,1-hf,params.F); %Filtering operator (1-hf) is the highpass filtering kernel
    
   
    
    smartinit = 0; 
    [xx,yy]=meshgrid(linspace(-ncols/2,ncols/2,ncols),linspace(-nrows/2,nrows/2,nrows));
    r = sqrt(xx.^2+yy.^2);
    mask = ifftshift(cast((r<205),'single'));
  
    init_eps = 1e-2;%Smart initialization regularization factor
    hipf = 1-hf; %This is the fourier transform of delta - hf filter
    a_gammaf = params.F*a_gamma;
    a_tkf0=(conj(hipf).*a_gammaf)./(abs(hipf).^2+init_eps);%Weiner deconvolution
    a_tk = params.F'*a_tkf0;       

    epochidx = 0;
    if (gpu_compute_en==0)
        for epochidx = 0:10
            disp(['Working at epoch ' num2str(epochidx)]);
            a_tk_new = estimate_gt_linear(a_gamma,hf,params,a_tk); %Solve with the linear model
            %Compute the FFT of the new image
            a_gammaf = fft2(a_gamma);
            a_tk_newf = fft2(a_tk_new);
            a_gammaf_shifted = fftshift(a_gammaf);
            a_tk_newf_shifted = fftshift(a_tk_newf);
            a_tk_newf2 = a_tk_newf.*(mask==0)+a_gammaf.*(mask==1);
            a_tk_newf2_shifted = fftshift(a_tk_newf2);
            figure(4);
            hold off;
            plot(log10(abs(a_gammaf_shifted(round(nrows/2),:))),'-r','LineWidth',1);
            hold on;
            plot(log10(abs(a_tk_newf_shifted(round(nrows/2),:))),'-b','LineWidth',1);
            
            hold on;
            plot(log10(abs(a_tk_newf2_shifted(round(nrows/2),:))),'-g','LineWidth',1);
            hold off;
            legend('Original','After loop 1','replaced');

            a_tk = real(ifft2(a_tk_newf2));
           
            figure(5);
            subplot(121);
            imagesc(a_gamma);colormap gray;
            title('Input image');
            subplot(122);
            imagesc(a_tk);colormap gray;
            title('Frequency replaced...');
       
        end
        
        %[tk] = estimate_gt(gamma,hf,params); %Non-linear solver
        
     else %Compute gk and tk on gpu
       % d = gpuDevice();
       % reset(d); %Reset the device and clear its memmory
       % [gk,tk] = estimate_gt_gpu(gamma,hf,params);
    end
    writeTIFF(tk,strcat(datafolder,filename,'_rec.tif'))
    disp('Done with solving the inverse problem....');
    