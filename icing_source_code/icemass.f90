SUBROUTINE icemass(w,T,P,Vmag,dt,Nd,Dstart,rho_it_start,icestart,icediam,lsnow,numnull)

  USE glkind, ONLY: jpim,jprb
  USE konstants, ONLY: epsilo,pi

  IMPLICIT NONE

  REAL(KIND=jprb), INTENT(IN   ) :: w, T, P, Vmag, dt, Nd, Dstart
  REAL(KIND=jprb), INTENT(INOUT ) :: rho_it_start, icestart, icediam
  LOGICAL, INTENT(IN   ) :: lsnow
  INTEGER(KIND=jpim), INTENT(INOUT ) :: numnull

  REAL(KIND=jprb) :: rho_w, alpha1min1, alpha1min2, Rd
  REAL(KIND=jprb) :: ka, lamb, Lf, Ts, Le, cp, cw, es_surf, r, sigma, Lk

  REAL (KIND=jprb) :: rho_a, mu
  REAL (KIND=jprb) :: D, Rho_ice, Rho_it, Mice, A, B, C, K, Phi, Red, ds, alpha1, alpha2, &
       alpha3, dMdt, Vo, Rv     
  REAL (KIND=jprb) :: Redd, Nu, h, F, esa
  REAL :: Rr, Ko

! Constants:
! w = Cloud liquid water content[g/m^3], time series
! T = Temperature in Celcius, time series
! P = Pressure in hektopascal, time series
! Vmag = Wind speed m/s, time series
! Mice = icemass in kg/m, time series

  rho_w = 1000._jprb             ! Density of water

  alpha1min1 = 0.01_jprb         ! lower limit (Finstad et al
                                 ! 1988, J.Oceanic Atmos. Technol., 5,160-170)
  alpha1min2 = 0.005_jprb        !  Absolute lower limit ??
  Rd = 287._jprb                 !  Gas constant for dry air
  Rv = 461.5_jprb                !  Gas constant for water vapor

!  alpha3-related constants

  ka = 0.02_jprb                 !  Thermal cond of air (W/mK)
  lamb = 0.3_jprb                !  liquid fraction during wet growth
  Lf = 333000._jprb              !  Latent heat released when freezing (kJ/kg)
  Ts = 0._jprb                   !  Surface temperature during wet growth
  Le = 2500000._jprb             !  Heat released during evaporation (2272 kJ/kg fra wikipedia)
  cp = 1004.3_jprb               !  Specific heat capacity (air)
  cw = 4200._jprb                !  Specific heat capacity (water)
  es_surf = 6.17_jprb            !  Saturation water vapor pressure at 0 deg C.
  r = 0.79_jprb                  !  Recovery factor for viscous heating (0.79 for sylinder)
  sigma = 5.67E-8_jprb           !  Stefan Boltzmanns constant
  Lk = 8.1E7_jprb                !  Linearis Konst. (K.Harstveit)

  D = icediam
  Rho_ice = 0._jprb

  A = 0._jprb
  B = 0._jprb
  C = 0._jprb
  K = 0._jprb
  Phi = 0._jprb
  Red = 0._jprb
  ds = 0._jprb
  alpha1 = 1._jprb
  alpha2 = 1._jprb

  IF(lsnow)alpha2 = 1._jprb/Vmag
  alpha2 = MIN(alpha2,1._jprb)

  alpha3 = 1._jprb

  dMdt = 0._jprb
  Rr = 0.
  Ko = 0.
  Vo = 0._jprb
  Redd = 0._jprb
  Nu = 0._jprb
  h = 0._jprb
  F = 0._jprb
  esa = 0._jprb

  Mice = icestart
  Rho_it = rho_it_start

!-------------------------------------------------------------------

!  Compute air density and viscosity

  rho_a = ((P*100._jprb)/((T+273.15_jprb)*Rd))
  mu = 1.718_jprb*1.E-5_jprb + 5.1_jprb*1.E-8_jprb*T
    
!  Compute alpha1
  
  ds = (((6._jprb*w)/(pi*Nd))**(1._jprb/3._jprb))*1.e2_jprb !  micro meter

  IF(.NOT. lsnow)THEN 
     CALL mvdcalc(w*1.e-3_jprb,Nd*1.e6_jprb,ds)
     ds = ds * 1.e6_jprb
  ENDIF

  !IF(.NOT. lsnow)ds = mvdcalc(w*1.e-3_jprb,Nd*1.e6_jprb)*1.e6_jprb

  Red = rho_a*ds*Vmag*1.e-6_jprb/mu

  K = (rho_w*((ds*1.e-6_jprb)**2.)*Vmag)/(9._jprb*mu*D)

  Phi = (Red**2._jprb)/K

  A = 1.066_jprb*(K**(-0.00616_jprb))*EXP(-1.103_jprb*(K**(-0.688_jprb)))  

  B = 3.641_jprb*(K**(-0.498_jprb))*EXP(-1.497_jprb*(K**(-0.694_jprb)))

  IF (Phi > 100._jprb) THEN
     C = 0.00637_jprb*((Phi-100._jprb)**0.381_jprb)
  ENDIF

  alpha1 = A-0.028_jprb-C*(B-0.0454_jprb)

  IF(alpha1 < alpha1min1) THEN
     alpha1 = alpha1min2/SQRT(D)
  ENDIF

!  Compute alpha3

  esa = 6.107_jprb*10._jprb**( (7.5_jprb*(T))/(237._jprb+T))

  Redd = rho_a*D*Vmag/mu

  Nu = 0.032_jprb*Redd**0.85_jprb

  h = ka*Nu/D

  F = alpha1*alpha2*w*Vmag*1.e-3_jprb

  alpha3 = h/(F*Lf*(1.-lamb))*( (Ts-T) + epsilo*Le/(cp*P)*(es_surf-esa)- &
       ((r*Vmag**2._jprb)/(2._jprb*cp)) )+(cw*(Ts-T))/((1._jprb-lamb)*Lf) + &
       (sigma*Lk*(Ts-T))/(F*Lf*(1._jprb-lamb) )

!db
!  write(*,*) 'alpha1,alpha2,alpha3=', alpha1,alpha2,alpha3
!db
  alpha3 = MIN(alpha3,1._jprb)
  alpha3 = MAX(alpha3,0._jprb)


!  --------------------------------------------

  dMdt = D*w*Vmag*alpha1*alpha2*alpha3

  Mice = Mice + dMdt*dt*1.e-3_jprb

! -------------------------------------------

  Rho_ice = 400._jprb

! -------------------------------------------
!  Update the cylinder diameter 

  D = SQRT((4._jprb*(dMdt*dt*1.e-3_jprb)/(Rho_ice*pi))+(D*D))

!  Update total ice density

!db
!  write(*,*) 'Mice, D, Dstart=', Mice, D, Dstart
!  write(*,*) w,T,P,Vmag,dt,Nd,Dstart
!  write(*,*) rho_it_start,icestart,icediam,lsnow
!  write(*,*)
!db

  if ( abs(D-Dstart) > 0.000001 ) then
     Rho_it = 4._jprb*Mice/(pi*((D*D)-(Dstart*Dstart)))
     D = MAX(D,Dstart)
     rho_it_start =  Rho_it
     icestart = Mice
     icediam = D
  else
     numnull = numnull + 1
!     write(*,*) 'Mice, D, Dstart=', Mice, D, Dstart
!     write(*,*) w,T,P,Vmag,dt,Nd,Dstart
!     write(*,*) rho_it_start,icestart,icediam,lsnow
  endif

  RETURN

END SUBROUTINE icemass

SUBROUTINE MVDCALC(lwc,Nt_c,mvd)

  USE glkind, ONLY: jprb
  USE konstants, ONLY: pi

  IMPLICIT NONE

  REAL(KIND=jprb), INTENT(IN) :: lwc,Nt_c
  REAL(KIND=jprb), INTENT(INOUT) :: mvd

  REAL(KIND=jprb) :: rho_w, am_r, bm_r, obmr, mu_c, cce2, ccg2, cce1, ccg1, ocg1
  REAL(KIND=jprb) :: lamc

  rho_w = 1000.0_jprb

  am_r = pi*rho_w/6.0_jprb
  bm_r = 3.0_jprb

  obmr =  1._jprb/bm_r

  mu_c = MIN(15._jprb, (1000._jprb*1.E6_jprb / Nt_c + 2._jprb))

  cce2 =  bm_r + mu_c + 1._jprb
  CALL wgamma(cce2,ccg2)

  cce1 =  mu_c + 1._jprb
  CALL wgamma(cce1,ccg1)

  ocg1 = 1._jprb/ccg1

  lamc = (Nt_c*am_r* ccg2 * ocg1/lwc)**obmr
  mvd = (3.0_jprb+mu_c+0.672_jprb)/lamc

  RETURN

END SUBROUTINE MVDCALC

SUBROUTINE WGAMMA(y,gamma)

  USE glkind, ONLY: jprb
  IMPLICIT NONE
  REAL(KIND=jprb), INTENT(IN) :: y
  REAL(KIND=jprb), INTENT(INOUT) :: gamma
  REAL(KIND=jprb) :: yy

  yy=y
  CALL gammln(yy)
  gamma = EXP(yy)

  RETURN

END SUBROUTINE WGAMMA

SUBROUTINE GAMMLN(XX)

!     --- RETURNS THE VALUE LN(GAMMA(XX)) FOR XX > 0.

  USE glkind, ONLY: jpim,jprb

  IMPLICIT NONE
  REAL(KIND=jprb), INTENT(INOUT) :: XX

  DOUBLE PRECISION, PARAMETER:: STP = 2.5066282746310005D0
  DOUBLE PRECISION, DIMENSION(6), PARAMETER :: &
       COF = (/76.18009172947146D0, -86.50532032941677D0, &
       24.01409824083091D0, -1.231739572450155D0, &
       .1208650973866179D-2, -.5395239384953D-5/)
  DOUBLE PRECISION :: SER,TMP,X,Y
  INTEGER(KIND=jpim):: J

  X = XX
  Y = X
  TMP = X+5.5D0
  TMP = (X+0.5D0)*LOG(TMP)-TMP
  SER = 1.000000000190015D0
  DO J = 1,6
     Y = Y+1.D0
     SER = SER+COF(J)/Y
  ENDDO

  XX = TMP+LOG(STP*SER/X)

  RETURN

END SUBROUTINE GAMMLN

SUBROUTINE MOMG(ZALPHA,ZNU,ZP,MOM) 

  USE glkind, ONLY: jprb

  IMPLICIT NONE
  REAL(KIND=jprb), INTENT(IN) :: ZALPHA,ZNU,ZP
  REAL(KIND=jprb), INTENT(INOUT) :: MOM

  REAL(KIND=jprb) :: W1,W2

  CALL wgamma(ZNU+ZP/ZALPHA,W1)
  CALL wgamma(ZNU,W2)
  
  MOM = W1/W2

  RETURN

END SUBROUTINE MOMG

SUBROUTINE LAMDA_R(RAIN,RHO,LAMDA)

  USE glkind, ONLY: jprb
  USE konstants, ONLY: pi

  IMPLICIT NONE
  REAL(KIND=jprb), INTENT(INOUT) :: LAMDA
  REAL(KIND=jprb), INTENT(IN) :: RAIN, RHO

  REAL(KIND=jprb) :: XRHOLW,XAR,XLBR,XLBEXR,XCCR,MOM,XALPHAR,XNUR,XBR

  XRHOLW = 1000._jprb

  XAR     = (PI/6.0_jprb)*XRHOLW
  XCCR    = 8.E6_jprb
  XALPHAR = 1.0_jprb
  XNUR    = 1.0_jprb
  XBR     = 3.0_jprb
  XLBEXR  = 1.0_jprb/(-1.0_jprb-XBR)

  CALL momg(XALPHAR,XNUR,XBR,MOM)
  XLBR   = ( XAR*XCCR*MOM )**(-XLBEXR)

  LAMDA = XLBR * ( RHO*RAIN ) ** XLBEXR

  RETURN

END SUBROUTINE LAMDA_R

SUBROUTINE LAMDA_S(SNOW,RHO,LAMDA)

  USE glkind, ONLY: jprb

  IMPLICIT NONE
  REAL(KIND=jprb), INTENT(INOUT) :: LAMDA
  REAL(KIND=jprb), INTENT(IN) :: SNOW, RHO

  REAL(KIND=jprb) :: XLBS,XLBEXS,XCXS,XAS,XCCS,MOM,XALPHAS,XNUS,XBS

  XAS     = 0.02_jprb
  XCCS    = 5.0_jprb
  XALPHAS = 1.0_jprb
  XNUS    = 1.0_jprb
  XBS     = 1.9_jprb
  XCXS    = 1.0_jprb
  XLBEXS  = 1.0_jprb/(XCXS-XBS)

  CALL momg(XALPHAS,XNUS,XBS,MOM)
  XLBS   = ( XAS*XCCS*MOM )**(-XLBEXS)

  LAMDA  = XLBS*( RHO*SNOW )**XLBEXS

  RETURN

END SUBROUTINE LAMDA_S

SUBROUTINE LAMDA_G(GRAUPEL,RHO,LAMDA)

  USE glkind, ONLY: jprb

  IMPLICIT NONE
  REAL(KIND=jprb), INTENT(INOUT) :: LAMDA
  REAL(KIND=jprb), INTENT(IN) :: GRAUPEL, RHO

  REAL(KIND=jprb) :: XLBG,XLBEXG,XAG,XCCG,XCXG,MOM,XALPHAG,XNUG,XBG

  XAG     = 19.6_jprb
  XCCG    = 5.E5_jprb
  XALPHAG = 1.0_jprb
  XNUG    = 1.0_jprb
  XBG     = 2.8_jprb
  XCXG    = -0.5_jprb
 
  XLBEXG = 1.0_jprb/(XCXG-XBG)

  CALL momg(XALPHAG,XNUG,XBG,MOM)
  XLBG = ( XAG*XCCG*MOM )**(-XLBEXG)

  LAMDA  = XLBG*( RHO*GRAUPEL )**XLBEXG

  RETURN

END SUBROUTINE LAMDA_G

