
    clc;
    close all;
    clear all;
    %Step1; Create a phantom objects
    Nx = 512;
    nrows = Nx;
    ncols = Nx;  
    
    %Step 2: compute the true formula T*(T v H)*
    htype = 'gaussian';
    
     switch (htype)
        case {'gaussian'} %A gaussian filter
                    bw = 30; %Bandwidth parameter
                    h=fspecial('gaussian',[round(4*bw)+1 round(4*bw)+1],bw); %Transfer function of the low-pass filter...
                    %Fourier transform the filter
                    h1 = zeros(nrows,ncols);
                    h1(1:size(h,1),1:size(h,2))=h;
                    kernel_size=size(h,1);
                    h1 = circshift(h1,[-round((kernel_size-1)/2) -round((kernel_size-1)/2)]);
                    hf = fft2(h1);
 
        case {'lp'} %Bandpass filter
                    lp_bw = 50;%Bandwidth of the low pass filter in the frequency domain. The smaller it is, the more coherent the field will be 
                    hf = zeros(nrows,ncols);
                    [x,y]=meshgrid(linspace(-ncols/2,ncols/2-1,ncols),linspace(-nrows/2,nrows/2-1,nrows));
                    mask = sqrt(x.^2+y.^2)<lp_bw; %
                    hf(mask)=1;
                    hf = fftshift(hf);
    end
       
   
    
    
    datasource = 1;
    if (datasource==1) %Generate simulation data
        phi = pi/2;
        amplitude = ones(nrows,ncols);
        phase = zeros(nrows,ncols);
        phase(round(nrows*0.35:nrows*0.65),round(ncols*0.35:ncols*0.65))=phi;
        amplitude(round(nrows*0.25:nrows*0.75),round(ncols*0.25:ncols*0.75))=1;
        T= amplitude.*exp(i*phase);
        nrows = size(T,1);
        ncols = size(T,2);
        x_arr = 1:ncols;
        y_arr = 1:nrows;
        [xx,yy]=meshgrid(x_arr,y_arr);
        slope = 0.000;
        phi_lin =exp(i*xx*slope);
        T = T.*phi_lin;
        Tf = fft2(T);
        loTf = Tf.*hf;
        loT = ifft2(loTf); 
        gamma=T.*conj(loT); %Generate simulation data for Gamma
        figure(1);colormap jet;
        subplot(221);imagesc(angle(T));axis off;colorbar;title('arg(T)');
        subplot(222);imagesc(angle(gamma));axis off;colorbar;title('arg(gamma)');
        subplot(223);imagesc(abs(T));axis off;colorbar;title('abs(T)');
        subplot(224);imagesc(abs(gamma));axis off;colorbar;title('abs(gamma)');drawnow;
        %Display the cross-section of the image
        mcs=angle(gamma(round((nrows+1)/2),:));%Get cross-section of the measurement
        cs = angle(T(round((nrows+1)/2),:));
        amp = abs(T(round((nrows+1)/2),:));%Amplitude of the T
        mamp = abs(gamma(round((nrows+1)/2),:));%Amplitude of the T
   
        figure(2);
        subplot(211);plot(mcs);hold on;plot(cs,'r');hold on;title('Cross sections for phase');
        legend('Gamma(r)','T(r)');drawnow;
        subplot(212);plot(mamp);hold on;plot(amp,'r');hold on;title('Cross sections for amplitude');
        legend('Gamma(r)','T(r)');drawnow;
    end
    if (gpuDeviceCount()==0)
        gpu_compute_en = 1; %if there is no gpu, compute the result on cpu instead
    end
    gpu_compute_en = 0;
    
    method = 'relax'; %Choose between the two: 'relax','cg','nlcg'
    
    %Parameter definitions
    params.niter = 1000; %Number of iterations needed
    params.lambda = 10;
    params.tol = 1e-5; %Tolerance for the solver to stop
    params.method = 'relax';%Choose between 'relax'/'cg'/'nlcf'
    params.smart_init_en = 1;
    %Operator definitions
    params.F = FFT2(Nx); %Fourier transform operator
    params.D = Grd(Nx);
    params.H = H(Nx,1-hf,params.F); %Filtering operator (1-hf) is the highpass filtering kernel
    
    
    smartinit = 0; 
    if (gpu_compute_en==0)
        [gk,tk] = estimate_gt(gamma,hf,params);
    else %Compute gk and tk on gpu
        d = gpuDevice();
        reset(d); %Reset the device and clear its memmory
        [gk,tk] = estimate_gt_gpu(gamma,hf,params);
    end
    disp('Done with solving the inverse problem....');
    