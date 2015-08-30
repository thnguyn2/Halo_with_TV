function [step,iter]=ls(y,u,pk,params,method)
%step=ls(xk,pk,params,method)
%This function perform line-searching algorithm using cubic interpolation
%to infer the suitable update step
%Input:
%   xk: current estimation
%   pk: current searching direction
%   params: paramters for searching
%   method:
%       'back': for back tracking line-search
%       'interp': for interpolation based line-search
%Output:
%   alpha: step size
%Reference:
%[1]. Nocedal, "Numerical Optimization", 2006 (chapter 3)
%Author: Tan H. Nguyen
%Version: 1.2
%The user need to define 2 function fval and gfval for computing values of
%f and its gradient
%=====================================================
    step=params.step0;
    c=params.LSc;
    phi0=fval(y,u,params); %Change fval, gfval according to different cases
    gphi0=gfval(y,u,params); %Gradient at 0-size step
    iter=0;
    switch method
        case {'back'} %Back tracking line-search
             phi_ak=fval(y,u+step*pk,params);
             %curthresh = phi0-c*step*abs(gphi0(:)'*pk(:)); %c = alpha
             %while ((phi_ak>curthresh)&(iter<params.LSMaxiter))
             while ((phi_ak>phi0)&(iter<params.LSMaxiter))              
                 step=step*params.LSrho;
                 phi_ak=fval(y,u+step*pk,params);
                 iter=iter+1;
                 %curthresh = phi0+c*step*abs(gphi0(:)'*pk(:));
             end
        case {'interp'} %Interpolation based line-search (see section 3.5 of [1])
            step0=step;
            dphi0=(gphi0(:))'*pk(:);%phi'(0)
            phi_a0=fval(y,u+step0*pk,params);%phi(alpha_0)
            if (phi_a0<phi0+c*step0*dphi0)%If initial condition is satisfied
               return;
            else
               iter=iter+1; 
               step1=-dphi0*step0^2/2/(phi_a0-phi0-dphi0*step0);
               %Minimum value of the interpolated quadratic function 
               phi_a1=fval(y,u+step1*pk,params);
               if (phi_a1<phi0+c*step1*dphi0) 
                    step=step1;
                    %disp('quad')
                    return;
               else
                    %Cubic polynormial interpolation
                    pre_step=step0; pre_phi=phi_a0;
                    cur_step=step1; cur_phi=phi_a1;                    
                    while (1)
                        b=1/pre_step^2/cur_step^2/(cur_step-pre_step)*...
                            [pre_step^2 -cur_step^2;-pre_step^3 cur_step^3]*...
                          [cur_phi-phi0-dphi0*cur_step;pre_phi-phi0-dphi0*pre_step];
                        step2=(-b(2)+sqrt(b(2)^2-3*b(1)*dphi0))/3/b(1);
                        r=step2/step1;
                        if (r>1)
                            disp('Error in estimating update step with cubic interpolation')
                            return
                        end
                        if ((r<0.01)||(r>0.99))
                            step2=cur_step/2;
                        end
                        phi_a2=fval(y,u+step2*pk,params);
                        if (phi_a2<phi0+c*cur_step*dphi0)
                            step=step2;
                            %disp(['Cubic: ' num2str(iter-1)]);
                            return;
                        end
                        %If not return then, update the step
                        pre_step=step1; pre_phi=phi_a1;
                        cur_step=step2; cur_phi=phi_a2;
                        iter=iter+1;
                    end
               end
            end
            %new_obj=fval(xk+step*pk,params);
           
            
        otherwise 
            error('Unknown line-search method')
    end
end