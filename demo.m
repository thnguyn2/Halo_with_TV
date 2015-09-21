
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
       filename = '10xPh1_03';
       a_gamma = -cast(imread(strcat('Pillars_data/',filename,'.tif')),'single');
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
    htype = 'measured';
    
     switch (htype)
        case {'gaussian'} %A gaussian filter
                    bw = 15; %Bandwidth parameter
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
    params.niter =200; %Number of iterations needed
    params.lambda = 1;
    params.beta = 1;
    params.tol = 1e-5; %Tolerance for the solver to stop
    params.method = 'relax';%Choose between 'relax'/'cg'/'nlcf'
    params.smart_init_en = 1;
    %Operator definitions
    params.F = FFT2(Nx); %Fourier transform operator
    params.TV = TV(); %Total variational prior
    params.H = H(Nx,1-hf,params.F); %Filtering operator (1-hf) is the highpass filtering kernel
    
   
    
    smartinit = 0; 
    if (gpu_compute_en==0)
        tk = estimate_gt_linear(a_gamma,hf,params); %Solve with the linear model
        %[tk] = estimate_gt(gamma,hf,params); %Non-linear solver
        
     else %Compute gk and tk on gpu
       % d = gpuDevice();
       % reset(d); %Reset the device and clear its memmory
       % [gk,tk] = estimate_gt_gpu(gamma,hf,params);
    end
    writeTIFF(tk,strcat('Pillars_data/',filename,'_rec.tif'))
    disp('Done with solving the inverse problem....');
    