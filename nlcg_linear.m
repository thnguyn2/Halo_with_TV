function x=nlcg_linear(y,params,x0)
%Non-linear conjugatte gradient with line-search
%x=argmin ||y-H*x||_2^2+ alpha*TV(x) =  argmin f(x)
%However, gradient of |V*u|_p norm can not be computed easily.
%Here lambda contains the trade off between data consistency and sparseness
%constrain.
%Author: Tan H. Nguyen
%Writing date: 10/24/2011
%Email: thnguyn2@illinois.edu
%University of Illinois at Urbana-Champaign
%--------------------------------------------------------------------------
%Arguments:
%Input:
%   x0: initial estimation of the recovered image
%   y: measured image
%   params: a struct containing all the constants and operators
%Reference:
%[1]. Nocedal, 'Numerical Optimization', 2006
%[2]. An introduction to the Conjugate Gradient Method without the
%Argonizaing Pain

%Remarks:
%Note that the second term of f is not differentiable. Thus, we replace it
%by sqrt(uT*u+muy) where muy is a very small number.
%--------------------------------------------------------------------------
[nr,nc]=size(x0);

%Parameters for Line-Search algorithm
params.LSc=0.01;%The constant c using in Line Search algorithm
                        %Note that the normalizing coefficients is needed
                        %here!!!
params.LSrho=0.6;%Rho used in Line Search algorithm
params.LSMaxiter=100; %Max number of iteration for line-search
params.step0=1;
params.CgTol=1e-5;%Tolerence on the gradient norm to stop iterating 


% figure(7);
% imagesc(y);
% title('Measured coefs');


%% Prepare for 1st iteration
[fk,dc,tv]=fval_linear(y,x0,params);%New objective
disp(['Initial Objectives:' ' Obj: ' num2str(fk,'%3.4f') ', dc: ' num2str(dc,'%0.3f'), ', TV: ' num2str(tv,'%0.3f')]);
gf0 =gfval_linear(y,x0,params);
%------------------------------------------------------------

disp('Nonlinear conjugate gradient method for optimization');
pk=-gf0;%Initial searching direction, pk is dx.
xk=x0;
gfk=gf0;

k=0;
obj_arr=zeros(0);
while ((k<params.niter)&&(norm(gfk,'fro')>params.CgTol))
           
        step = params.step0; 
        [f0,dc,tv] = fval_linear(y, xk, params);
        
                  
        k=k+1;
        disp(['#' num2str(k) ', step: ' num2str(step) ...
                ', Obj: ' num2str(f0,'%0.5f') ', dc:' num2str(dc,'%0.5f')...
                ', TV:' num2str(tv,'%0.5f')]);

        
        
        f1 = fval_linear(y, xk+step*pk, params);
        lsiter = 0;
        while (f1 > f0 - params.LSc*step*abs(gfk(:)'*pk(:))) & (lsiter<params.LSMaxiter)%alpha = 0.01, t0 =1
        	lsiter = lsiter + 1;
            step = step * params.LSrho;
            f1 = fval_linear(y, xk+step*pk, params);
        end
         
         if (lsiter==0)
            params.step0=params.step0/params.LSrho; %If previous step can be found very quickly then increase step size
         else
            params.step0=params.step0*params.LSrho; %reduce intial searching step with knowledge given in previous step
         end
         %State update
         xk=xk+step*pk;

         %Calculate new gradient
         gfk1=gfval_linear(y,xk,params);

         %Updating coefficients
         beta=(gfk1(:)'*gfk1(:))/(gfk(:)'*gfk(:)+eps);
         gfk = gfk1;
         pk=-gfk1+beta*pk;
         
    
         if (mod(k,1000)==0)
             pk = -gfk1; %Restart
         end
          
%            figure(2);
%             hold off;
%             plot(x0(980,:),'b');
%             hold on;
%             plot(xk(980,:),'r');
%             hold off;
%             legend('Original','Reconstructed');
%             %colormap jet;
%             drawnow;
%             figure(3);
%             imagesc(xk);colormap gray;colorbar
%            
        
         
      
    end
    x=xk;
  
    




