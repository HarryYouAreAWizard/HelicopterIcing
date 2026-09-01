SUBROUTINE icing3(par_ind,nreadkey,levtop,readkey,geo,allfld)

 ! 
 ! calculate maximum icing index based on icegrowth on a cylinder
 ! using all cloud species
 !
 ! Esbj�rn Olsson SMHI, nov 2013

 USE glkind,           ONLY : jpim,jprb
 USE typez,            ONLY : geometry,grib_api_key
 USE grib_api_list,    ONLY : soft_check_key,init_grib_api_key
 USE konstants,        ONLY : zz,pfull,do_icing,tzero
 USE namelist_control, ONLY : rmisval

 IMPLICIT NONE

 INTEGER(KIND=jpim), PARAMETER :: maxpar = 7
 CHARACTER(LEN=9),   PARAMETER :: pname(maxpar) = &
    (/'t        ','q        ','ciwc_cond','cwat_cond', &
      'rain_cond','snow_cond','grpl_cond'/)

 ! Input
 INTEGER(KIND=jpim), INTENT(IN   ) :: par_ind,nreadkey,levtop
 TYPE(grib_api_key), INTENT(IN   ) :: readkey(nreadkey)
 TYPE(geometry),     INTENT(IN   ) :: geo
 REAL(KIND=jprb),    TARGET,    INTENT(INOUT) :: allfld(geo%nlon,geo%nlat,nreadkey)

 ! Local
 INTEGER(KIND=jpim) :: i,j,jx,jy,jk,modlev(geo%nlon,geo%nlat)
 INTEGER(KIND=jpim) :: numnull

 REAL(KIND=jprb) :: Dstart, dt, ice_start, rho_it_start, weps, lwc, Nd, ff, ice_diam
 REAL(KIND=jprb) :: xccr, xccs, xccg, xcxs, xcxg, zfact_nucl, xalpi, xbetai, xgami, xalpw, &
                    xbetaw, xgamw, xmd, xmv, xtt, xnu10, xnu20, xalpha1, xalpha2, xbeta1, xbeta2, &
                    pcit, zcit, zzw, zssi, zusw
 REAL(KIND=jprb) :: lamdar, lamdas, lamdag

 LOGICAL :: lsnow
 REAL(KIND=jprb), SAVE, ALLOCATABLE, DIMENSION(:,:)   :: max_growth, maxlev, base, top
 REAL(KIND=jprb), SAVE, ALLOCATABLE, DIMENSION(:,:,:) :: icingind,ice_growth

 REAL(KIND=jprb), TARGET    :: &
                        cw(geo%nlon,geo%nlat,levtop:geo%nlev),     &
                      rain(geo%nlon,geo%nlat,levtop:geo%nlev),     &
                        ci(geo%nlon,geo%nlat,levtop:geo%nlev),     &
                      snow(geo%nlon,geo%nlat,levtop:geo%nlev),     &
                   graupel(geo%nlon,geo%nlat,levtop:geo%nlev),     &
                         t(geo%nlon,geo%nlat,levtop:geo%nlev),     &
                         q(geo%nlon,geo%nlat,levtop:geo%nlev),     &
                         p(geo%nlon,geo%nlat,levtop:geo%nlev),     &
                       rho(geo%nlon,geo%nlat,levtop:geo%nlev)

 TYPE(grib_api_key)  :: key
 REAL(KIND=jprb), POINTER :: fpoint(:,:)

 !
 ! ---------------------------------------------------------------------------
 !

 numnull = 0

 CALL init_grib_api_key(key)

 IF(do_icing)THEN

   IF (ALLOCATED(max_growth)) DEALLOCATE(max_growth,maxlev,base,top, &
                                         icingind,ice_growth)
   ALLOCATE(max_growth(geo%nlon,geo%nlat),              &
                maxlev(geo%nlon,geo%nlat),              &
                  base(geo%nlon,geo%nlat),              &
                   top(geo%nlon,geo%nlat),              &
              icingind(geo%nlon,geo%nlat,levtop:geo%nlev), &
              ice_growth(geo%nlon,geo%nlat,levtop:geo%nlev))

    xccr = 8.e6_jprb
    xccs = 5.0_jprb
    xccg = 5.e5_jprb
    xcxs = 1.0_jprb
    xcxg = -0.5_jprb
    zfact_nucl = 1.0_jprb
    xalpi = 0.3262E+02_jprb
    xbetai = 0.6295E+04_jprb
    xgami = 0.5631_jprb
    xalpw = 0.6022E+02_jprb
    xbetaw = 0.6822E+04_jprb
    xgamw = 0.5139E+01_jprb
    xmd = 0.2896E-01_jprb
    xmv = 0.1802E-01_jprb
    xtt = 0.2732E+03_jprb
    xnu10 = 50._jprb*zfact_nucl
    xnu20 = 1000._jprb*zfact_nucl
    xalpha1 = 4.5_jprb
    xalpha2 = 12.96_jprb
    xbeta1 = 0.6_jprb
    xbeta2 = 0.639_jprb

    key%levtype = 'hybrid'
    key%tri     = 000

    DO j = 1,maxpar
     DO i = levtop,geo%nlev

       key%level = i
       key%shortname = TRIM(pname(j))

       SELECT CASE(key%shortname)
       CASE('t')
        fpoint => t(:,:,i)
       CASE('q')
        fpoint => q(:,:,i)
       CASE('ciwc_cond')
        fpoint => ci(:,:,i)
       CASE('cwat_cond')
        fpoint => cw(:,:,i)
       CASE('rain_cond')
        fpoint => rain(:,:,i)
       CASE('snow_cond')
        fpoint => snow(:,:,i)
       CASE('grpl_cond')
        fpoint => graupel(:,:,i)
       END SELECT

       CALL soft_check_key(key,readkey,nreadkey,.TRUE.)
       IF ( key%pos /= -1 ) THEN
         fpoint = allfld(:,:,key%pos)
       ELSE
         WRITE(6,*)'Missing:',key%shortname,key%level,key%levtype
         CALL abort
       ENDIF

     ENDDO
    ENDDO


    DO jk = levtop,geo%nlev
       rho(:,:,jk) = pfull(:,:,jk)/(t(:,:,jk) * 287._jprb)
    ENDDO

    icingind(:,:,:) = 0._jprb
      maxlev(:,:)   = rmisval
        base(:,:)   = rmisval
         top(:,:)   = rmisval
    max_growth(:,:) = 0._jprb
    modlev(:,:) = 0

    t = t - 273.15_jprb
    p(:,:,:) = pfull(:,:,levtop:geo%nlev) * 0.01_jprb

    WHERE(cw      < 1.e-10_jprb ) cw = 0.0_jprb
    WHERE(ci      < 1.e-10_jprb ) ci = 0.0_jprb
    WHERE(rain    < 1.e-10_jprb ) rain = 0.0_jprb
    WHERE(snow    < 1.e-10_jprb ) snow = 0.0_jprb
    WHERE(graupel < 1.e-10_jprb ) graupel = 0.0_jprb

    Dstart = 0.15_jprb
    dt = 3600._jprb
    weps = 1.e-8_jprb
    ff = 90._jprb ! 175 knots

!$omp parallel do private(jy,jx,jk,ice_start,rho_it_start,ice_diam,lwc,Nd,lsnow) &
!$omp& private(pcit,zcit,zzw,zssi,zusw)
    DO jk = levtop,geo%nlev
 
       DO jy = 1,geo%nlat

          DO jx = 1,geo%nlon

             IF (t(jx,jy,jk) < 0._jprb) THEN

                ice_start = 0._jprb
                rho_it_start = 100._jprb
                ice_diam = Dstart

                lwc = rho(jx,jy,jk)*1000._jprb*cw(jx,jy,jk)

                IF( lwc > 0._jprb)THEN
                
                   Nd = 100._jprb
                   lsnow = .FALSE.

                   CALL icemass(lwc,t(jx,jy,jk),p(jx,jy,jk),ff,dt,Nd,Dstart,rho_it_start,ice_start, &
                        ice_diam,lsnow,numnull)

                ENDIF

                lwc = rho(jx,jy,jk)*1000._jprb*rain(jx,jy,jk)

                IF( lwc > 0._jprb)THEN
                
                   CALL lamda_r(rain(jx,jy,jk),rho(jx,jy,jk),lamdar)
                   Nd = xccr*lamdar**(-1)
                   Nd = Nd * 1.e-6_jprb
                   lsnow = .FALSE.

                   CALL icemass(lwc,t(jx,jy,jk),p(jx,jy,jk),ff,dt,Nd,Dstart,rho_it_start,ice_start, &
                        ice_diam,lsnow,numnull)

                ENDIF

                lwc = rho(jx,jy,jk)*1000._jprb*ci(jx,jy,jk)

                IF( lwc > 0._jprb .AND. (cw(jx,jy,jk) > 0._jprb .OR. rain(jx,jy,jk) > 0._jprb) )THEN

                   pcit = 0._jprb

                   zcit = pcit

                   zzw = EXP( xalpi - xbetai/(t(jx,jy,jk)+xtt) - xgami*LOG((t(jx,jy,jk)+xtt)) )
                   zssi = q(jx,jy,jk)*( p(jx,jy,jk)*100._jprb-zzw ) / ( (xmv/xmd) * zzw ) - 1._jprb
                   zusw = EXP( xalpw - xbetaw/(t(jx,jy,jk)+xtt) - xgamw*LOG((t(jx,jy,jk)+xtt)) )
       
                   zzw = 0._jprb
                   zssi = MIN( zssi, zusw )

                   IF( (t(jx,jy,jk) < -5.0_jprb) .AND. (zssi > 0._jprb) ) THEN      
                      zzw = xnu20*EXP( xalpha2*zssi-xbeta2 )
                   END IF

                   IF( (t(jx,jy,jk) <= -2.0_jprb) .AND. (t(jx,jy,jk) >= -5.0_jprb) &
                        .AND. (zssi > 0.0) ) THEN
                      zzw = MAX( xnu20*EXP(-xbeta2), xnu10*EXP(-xbeta1*(t(jx,jy,jk)) )* &
                           ( zssi/zusw )**xalpha1 )
                   END IF

                   zzw = zzw - zcit

                   IF ( zzw > 0.0 ) THEN
                      zzw = MIN(zzw,50E3_jprb)
                      zzw = MAX( zzw + zcit, zcit )
                      pcit = zzw
                   END IF

                   Nd = pcit * 1.e-6_jprb
                   Nd = MAX(Nd,1.e-4_jprb)
                   IF(Nd == 0._jprb)Nd = 1.e-2_jprb

                   lsnow = .TRUE.

                   CALL icemass(lwc,t(jx,jy,jk),p(jx,jy,jk),ff,dt,Nd,Dstart,rho_it_start,ice_start, &
                        ice_diam,lsnow,numnull)

                ENDIF

                lwc = rho(jx,jy,jk)*1000._jprb*snow(jx,jy,jk)

                IF( lwc > 0._jprb .AND. (cw(jx,jy,jk) > 0._jprb .OR. rain(jx,jy,jk) > 0._jprb))THEN
                
                   CALL lamda_s(snow(jx,jy,jk),rho(jx,jy,jk),lamdas)
                   Nd = xccs*lamdas**xcxs
                   Nd = Nd * 1.e-6_jprb
                   Nd = MAX(Nd,5.e-2_jprb)
                   lsnow = .TRUE.

                   CALL icemass(lwc,t(jx,jy,jk),p(jx,jy,jk),ff,dt,Nd,Dstart,rho_it_start,ice_start, &
                        ice_diam,lsnow,numnull)

                ENDIF

                lwc = rho(jx,jy,jk)*1000._jprb*graupel(jx,jy,jk)

                IF( lwc > 0._jprb .AND. (cw(jx,jy,jk) > 0._jprb .OR. rain(jx,jy,jk) > 0._jprb) )THEN
                
                   CALL lamda_g(graupel(jx,jy,jk),rho(jx,jy,jk),lamdag)
                   Nd = xccg*lamdag**xcxg
                   Nd = Nd * 1.e-6_jprb
                   Nd = MAX(Nd,5.e-2_jprb)
                   lsnow = .TRUE.

                   CALL icemass(lwc,t(jx,jy,jk),p(jx,jy,jk),ff,dt,Nd,Dstart,rho_it_start,ice_start, &
                        ice_diam,lsnow,numnull)

                ENDIF

                ice_growth(jx,jy,jk) = ice_diam - Dstart

                IF (ice_growth(jx,jy,jk) > 0.075_jprb) THEN
                   icingind(jx,jy,jk) = 4._jprb
                ELSEIF (ice_growth(jx,jy,jk) > 0.025_jprb) THEN
                   icingind(jx,jy,jk) = 3._jprb
                ELSEIF (ice_growth(jx,jy,jk) > 0.00625_jprb) THEN
                   icingind(jx,jy,jk) = 2._jprb
                ELSEIF (ice_growth(jx,jy,jk) > 0.001_jprb) THEN
                   icingind(jx,jy,jk) = 1._jprb
                ENDIF

                IF(ice_growth(jx,jy,jk) > max_growth(jx,jy) .AND. icingind(jx,jy,jk) > 0._jprb)THEN
                   max_growth(jx,jy) = ice_growth(jx,jy,jk)
                   maxlev(jx,jy) = zz(jx,jy,jk)
                   modlev(jx,jy) = jk
                ENDIF

             ENDIF

          ENDDO
       
       ENDDO

    ENDDO
!$omp end parallel do

!$omp parallel do private(jy,jx,jk)
    DO jy = 1,geo%nlat

       DO jx = 1,geo%nlon

          max_growth(jx,jy) = 0._jprb

          IF (modlev(jx,jy) > 0) THEN

             jk = modlev(jx,jy)

             DO WHILE(icingind(jx,jy,jk) > 2._jprb .AND. jk < geo%nlev)
                base(jx,jy) = zz(jx,jy,jk)
                jk = jk + 1
             ENDDO

             jk = modlev(jx,jy)

             DO WHILE(icingind(jx,jy,jk) > 2._jprb .AND. jk >= levtop)
                top(jx,jy) = zz(jx,jy,jk)
                jk = jk - 1
             ENDDO

             max_growth(jx,jy) = icingind(jx,jy,modlev(jx,jy))

          ENDIF

       ENDDO

    ENDDO
!$omp end parallel do

    do_icing = .FALSE.

 ENDIF

 SELECT CASE(readkey(par_ind)%shortname)

   CASE('atmiceg')
    IF ( readkey(par_ind)%levtype == 'hybrid' ) THEN
      ! Scale to m/s
      allfld(:,:,par_ind) = ice_growth(:,:,readkey(par_ind)%level)/3600._jprb
    ELSE
      allfld(:,:,par_ind) = rmisval
    ENDIF
   CASE('icei2')
    IF ( readkey(par_ind)%levtype == 'heightAboveGround' ) THEN
      CALL icing2(par_ind,nreadkey,readkey,geo,allfld,icingind)
    ELSE
      allfld(:,:,par_ind) = icingind(:,:,readkey(par_ind)%level)
    ENDIF
   CASE('lmxice')
    allfld(:,:,par_ind) = maxlev(:,:)
   CASE('mxicegr')
    allfld(:,:,par_ind) = max_growth(:,:)
   CASE('blice')
    allfld(:,:,par_ind) = base(:,:)
   CASE('tlice')
    allfld(:,:,par_ind) = top(:,:)
   CASE DEFAULT
    allfld(:,:,par_ind) = rmisval

 END SELECT

 RETURN

END SUBROUTINE icing3
