!> Provides routines that do checksums of groups of MOM variables
module MOM_checksum_packages

! This file is part of MOM6. See LICENSE.md for the license.

!   This module provides several routines that do check-sums of groups
! of variables in the various dynamic solver routines.

use MOM_coms, only : min_across_PEs, max_across_PEs, reproducing_sum
use MOM_debugging, only : hchksum, uvchksum
use MOM_error_handler, only : MOM_mesg, is_root_pe
use MOM_grid, only : ocean_grid_type
use MOM_unit_scaling, only : unit_scale_type
use MOM_variables, only : thermo_var_ptrs, surface
use MOM_verticalGrid, only : verticalGrid_type

implicit none ; private

public MOM_state_chksum, MOM_thermo_chksum, MOM_accel_chksum
public MOM_state_stats, MOM_surface_chksum

!> Write out checksums of the MOM6 state variables
interface MOM_state_chksum
  module procedure MOM_state_chksum_5arg
  module procedure MOM_state_chksum_3arg
end interface

#include <MOM_memory.h>

!> A type for storing statistica about a variable
type :: stats ; private
  real :: minimum = 1.E34  !< The minimum value [degC] or [ppt] or other units
  real :: maximum = -1.E34 !< The maximum value [degC] or [ppt] or other units
  real :: average = 0.     !< The average value [degC] or [ppt] or other units
end type stats

contains

! =============================================================================

!> Write out chksums for the model's basic state variables, including transports.
subroutine MOM_state_chksum_5arg(mesg, u, v, h, uh, vh, G, GV, US, haloshift, symmetric, omit_corners, vel_scale)
  character(len=*),                          &
                           intent(in) :: mesg !< A message that appears on the chksum lines.
  type(ocean_grid_type),   intent(in) :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV   !< The ocean's vertical grid structure.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                           intent(in) :: u    !< The zonal velocity [L T-1 ~> m s-1] or other units.
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                           intent(in) :: v    !< The meridional velocity [L T-1 ~> m s-1] or other units.
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
                           intent(in) :: h    !< Layer thicknesses [H ~> m or kg m-2].
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                           intent(in) :: uh   !< Volume flux through zonal faces = u*h*dy
                                              !! [H L2 T-1 ~> m3 s-1 or kg s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                           intent(in) :: vh   !< Volume flux through meridional faces = v*h*dx
                                              !! [H L2 T-1 ~> m3 s-1 or kg s-1].
  type(unit_scale_type),   intent(in) :: US   !< A dimensional unit scaling type
  integer,       optional, intent(in) :: haloshift !< The width of halos to check (default 0).
  logical,       optional, intent(in) :: symmetric !< If true, do checksums on the fully symmetric
                                                   !! computational domain.
  logical,       optional, intent(in) :: omit_corners !< If true, avoid checking diagonal shifts
  real,          optional, intent(in) :: vel_scale !< The scaling factor to convert velocities to [T m L-1 s-1 ~> 1]

  real :: scale_vel ! The scaling factor to convert velocities to mks units [T m L-1 s-1 ~> 1]
  logical :: sym
  integer :: hs

  ! Note that for the chksum calls to be useful for reproducing across PE
  ! counts, there must be no redundant points, so all variables use is..ie
  ! and js...je as their extent.
  hs = 1 ; if (present(haloshift)) hs=haloshift
  sym = .false. ; if (present(symmetric)) sym=symmetric
  scale_vel = US%L_T_to_m_s ; if (present(vel_scale)) scale_vel = vel_scale

  call uvchksum(mesg//" [uv]", u, v, G%HI, haloshift=hs, symmetric=sym, &
                omit_corners=omit_corners, unscale=scale_vel)
  call hchksum(h, mesg//" h", G%HI, haloshift=hs, omit_corners=omit_corners, unscale=GV%H_to_MKS)
  call uvchksum(mesg//" [uv]h", uh, vh, G%HI, haloshift=hs, symmetric=sym, &
                omit_corners=omit_corners, unscale=GV%H_to_MKS*US%L_to_m**2*US%s_to_T)
end subroutine MOM_state_chksum_5arg

! =============================================================================

!> Write out chksums for the model's basic state variables.
subroutine MOM_state_chksum_3arg(mesg, u, v, h, G, GV, US, haloshift, symmetric, omit_corners)
  character(len=*),                intent(in) :: mesg !< A message that appears on the chksum lines.
  type(ocean_grid_type),           intent(in) :: G  !< The ocean's grid structure.
  type(verticalGrid_type),         intent(in) :: GV !< The ocean's vertical grid structure.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                                   intent(in) :: u  !< Zonal velocity [L T-1 ~> m s-1] or [m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                                   intent(in) :: v  !< Meridional velocity [L T-1 ~> m s-1] or [m s-1]..
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
                                   intent(in) :: h  !< Layer thicknesses [H ~> m or kg m-2].
  type(unit_scale_type),           intent(in) :: US !< A dimensional unit scaling type, which is
                                                    !! used to rescale u and v if present.
  integer,               optional, intent(in) :: haloshift !< The width of halos to check (default 0).
  logical,               optional, intent(in) :: symmetric !< If true, do checksums on the fully
                                                    !! symmetric computational domain.
  logical,               optional, intent(in) :: omit_corners !< If true, avoid checking diagonal shifts

  integer :: hs
  logical :: sym

  ! Note that for the chksum calls to be useful for reproducing across PE
  ! counts, there must be no redundant points, so all variables use is..ie
  ! and js...je as their extent.
  hs = 1 ; if (present(haloshift)) hs = haloshift
  sym = .false. ; if (present(symmetric)) sym = symmetric
  call uvchksum(mesg//" u", u, v, G%HI, haloshift=hs, symmetric=sym, &
                omit_corners=omit_corners, unscale=US%L_T_to_m_s)
  call hchksum(h, mesg//" h",G%HI, haloshift=hs, omit_corners=omit_corners, unscale=GV%H_to_MKS)
end subroutine MOM_state_chksum_3arg

! =============================================================================

!> Write out chksums for the model's thermodynamic state variables.
subroutine MOM_thermo_chksum(mesg, tv, G, US, haloshift, omit_corners)
  character(len=*),         intent(in) :: mesg !< A message that appears on the chksum lines.
  type(thermo_var_ptrs),    intent(in) :: tv   !< A structure pointing to various
                                               !! thermodynamic variables.
  type(ocean_grid_type),    intent(in) :: G    !< The ocean's grid structure.
  type(unit_scale_type),    intent(in) :: US   !< A dimensional unit scaling type
  integer,        optional, intent(in) :: haloshift !< The width of halos to check (default 0).
  logical,        optional, intent(in) :: omit_corners !< If true, avoid checking diagonal shifts

  integer :: hs
  hs=1 ; if (present(haloshift)) hs=haloshift

  if (associated(tv%T)) &
    call hchksum(tv%T, mesg//" T", G%HI, haloshift=hs, omit_corners=omit_corners, unscale=US%C_to_degC)
  if (associated(tv%S)) &
    call hchksum(tv%S, mesg//" S", G%HI, haloshift=hs, omit_corners=omit_corners, unscale=US%S_to_ppt)
  if (associated(tv%frazil)) &
    call hchksum(tv%frazil, mesg//" frazil", G%HI, haloshift=hs, omit_corners=omit_corners, &
                 unscale=US%Q_to_J_kg*US%R_to_kg_m3*US%Z_to_m)
  if (associated(tv%salt_deficit)) &
    call hchksum(tv%salt_deficit, mesg//" salt deficit", G%HI, haloshift=hs, omit_corners=omit_corners, &
                 unscale=US%S_to_ppt*US%RZ_to_kg_m2)
  if (associated(tv%varT)) &
    call hchksum(tv%varT, mesg//" varT", G%HI, haloshift=hs, omit_corners=omit_corners, unscale=US%C_to_degC**2)
  if (associated(tv%varS)) &
    call hchksum(tv%varS, mesg//" varS", G%HI, haloshift=hs, omit_corners=omit_corners, unscale=US%S_to_ppt**2)
  if (associated(tv%covarTS)) &
    call hchksum(tv%covarTS, mesg//" covarTS", G%HI, haloshift=hs, omit_corners=omit_corners, &
                 unscale=US%S_to_ppt*US%C_to_degC)

end subroutine MOM_thermo_chksum

! =============================================================================

!> Write out chksums for the ocean surface variables.
subroutine MOM_surface_chksum(mesg, sfc_state, G, US, haloshift, symmetric)
  character(len=*),      intent(in)    :: mesg !< A message that appears on the chksum lines.
  type(surface),         intent(inout) :: sfc_state !< transparent ocean surface state structure
                                               !! shared with the calling routine data in this
                                               !! structure is intent out.
  type(ocean_grid_type), intent(in)    :: G    !< The ocean's grid structure.
  type(unit_scale_type), intent(in)    :: US    !< A dimensional unit scaling type
  integer,     optional, intent(in)    :: haloshift !< The width of halos to check (default 0).
  logical,     optional, intent(in)    :: symmetric !< If true, do checksums on the fully symmetric
                                               !! computational domain.

  integer :: hs
  logical :: sym

  sym = .false. ; if (present(symmetric)) sym = symmetric
  hs = 0 ; if (present(haloshift)) hs = haloshift

  if (allocated(sfc_state%SST)) call hchksum(sfc_state%SST, mesg//" SST", G%HI, haloshift=hs, &
                                             unscale=US%C_to_degC)
  if (allocated(sfc_state%SSS)) call hchksum(sfc_state%SSS, mesg//" SSS", G%HI, haloshift=hs, &
                                             unscale=US%S_to_ppt)
  if (allocated(sfc_state%sea_lev)) call hchksum(sfc_state%sea_lev, mesg//" sea_lev", G%HI, &
                                                 haloshift=hs, unscale=US%Z_to_m)
  if (allocated(sfc_state%Hml)) call hchksum(sfc_state%Hml, mesg//" Hml", G%HI, haloshift=hs, &
                                             unscale=US%Z_to_m)
  if (allocated(sfc_state%u) .and. allocated(sfc_state%v)) &
    call uvchksum(mesg//" SSU", sfc_state%u, sfc_state%v, G%HI, haloshift=hs, symmetric=sym, &
                  unscale=US%L_T_to_m_s)
  if (allocated(sfc_state%frazil)) call hchksum(sfc_state%frazil, mesg//" frazil", G%HI, &
                                                haloshift=hs, unscale=US%Q_to_J_kg*US%RZ_to_kg_m2)
  if (allocated(sfc_state%melt_potential)) call hchksum(sfc_state%melt_potential, mesg//" melt_potential", &
                      G%HI, haloshift=hs, unscale=US%Q_to_J_kg*US%RZ_to_kg_m2)
  if (allocated(sfc_state%ocean_mass)) call hchksum(sfc_state%ocean_mass, mesg//" ocean_mass", &
                      G%HI, haloshift=hs, unscale=US%RZ_to_kg_m2)
  if (allocated(sfc_state%ocean_heat)) call hchksum(sfc_state%ocean_heat, mesg//" ocean_heat", &
                      G%HI, haloshift=hs, unscale=US%C_to_degC*US%RZ_to_kg_m2)
  if (allocated(sfc_state%ocean_salt)) call hchksum(sfc_state%ocean_salt, mesg//" ocean_salt", &
                      G%HI, haloshift=hs, unscale=US%S_to_ppt*US%RZ_to_kg_m2)

end subroutine MOM_surface_chksum

! =============================================================================

!> Write out chksums for the model's accelerations
subroutine MOM_accel_chksum(mesg, CAu, CAv, PFu, PFv, diffu, diffv, G, GV, US, pbce, &
                            u_accel_bt, v_accel_bt, symmetric)
  character(len=*),         intent(in) :: mesg !< A message that appears on the chksum lines.
  type(ocean_grid_type),    intent(in) :: G    !< The ocean's grid structure.
  type(verticalGrid_type),  intent(in) :: GV   !< The ocean's vertical grid structure.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                            intent(in) :: CAu  !< Zonal acceleration due to Coriolis
                                               !! and momentum advection terms [L T-2 ~> m s-2].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                            intent(in) :: CAv  !< Meridional acceleration due to Coriolis
                                               !! and momentum advection terms [L T-2 ~> m s-2].
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                            intent(in) :: PFu  !< Zonal acceleration due to pressure gradients
                                               !! (equal to -dM/dx) [L T-2 ~> m s-2].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                            intent(in) :: PFv  !< Meridional acceleration due to pressure gradients
                                               !! (equal to -dM/dy) [L T-2 ~> m s-2].
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                            intent(in) :: diffu !< Zonal acceleration due to convergence of the
                                                !! along-isopycnal stress tensor [L T-2 ~> m s-2].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                            intent(in) :: diffv !< Meridional acceleration due to convergence of
                                                !! the along-isopycnal stress tensor [L T-2 ~> m s-2].
  type(unit_scale_type),    intent(in) :: US    !< A dimensional unit scaling type
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
                  optional, intent(in) :: pbce !< The baroclinic pressure anomaly in each layer
                                               !! due to free surface height anomalies
                                               !! [L2 T-2 H-1 ~> m s-2 or m4 s-2 kg-1].
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                  optional, intent(in) :: u_accel_bt !< The zonal acceleration from terms in the
                                                     !! barotropic solver [L T-2 ~> m s-2].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                  optional, intent(in) :: v_accel_bt !< The meridional acceleration from terms in
                                                     !! the barotropic solver [L T-2 ~> m s-2].
  logical,        optional, intent(in) :: symmetric !< If true, do checksums on the fully symmetric
                                                    !! computational domain.

  logical :: sym

  sym=.false.; if (present(symmetric)) sym=symmetric

  ! Note that for the chksum calls to be useful for reproducing across PE
  ! counts, there must be no redundant points, so all variables use is..ie
  ! and js...je as their extent.
  call uvchksum(mesg//" CA[uv]", CAu, CAv, G%HI, haloshift=0, symmetric=sym, unscale=US%L_T2_to_m_s2)
  call uvchksum(mesg//" PF[uv]", PFu, PFv, G%HI, haloshift=0, symmetric=sym, unscale=US%L_T2_to_m_s2)
  call uvchksum(mesg//" diffu", diffu, diffv, G%HI,haloshift=0, symmetric=sym, unscale=US%L_T2_to_m_s2)
  if (present(pbce)) &
    call hchksum(pbce, mesg//" pbce",G%HI,haloshift=0, unscale=GV%m_to_H*US%L_T_to_m_s**2)
  if (present(u_accel_bt) .and. present(v_accel_bt)) &
    call uvchksum(mesg//" [uv]_accel_bt", u_accel_bt, v_accel_bt, G%HI,haloshift=0, symmetric=sym, &
                  unscale=US%L_T2_to_m_s2)
end subroutine MOM_accel_chksum

! =============================================================================

!> Monitor and write out statistics for the model's state variables.
subroutine MOM_state_stats(mesg, u, v, h, Temp, Salt, G, GV, US, allowChange, permitDiminishing)
  type(ocean_grid_type),   intent(in) :: G    !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV   !< The ocean's vertical grid structure.
  character(len=*),        intent(in) :: mesg !< A message that appears on the chksum lines.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                           intent(in) :: u    !< The zonal velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                           intent(in) :: v    !< The meridional velocity [L T-1 ~> m s-1].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  &
                           intent(in) :: h    !< Layer thicknesses [H ~> m or kg m-2].
  real, pointer, dimension(:,:,:),           &
                           intent(in) :: Temp !< Temperature [C ~> degC].
  real, pointer, dimension(:,:,:),           &
                           intent(in) :: Salt !< Salinity [S ~> ppt].
  type(unit_scale_type),   intent(in) :: US    !< A dimensional unit scaling type
  logical,       optional, intent(in) :: allowChange !< do not flag an error
                                                     !! if the statistics change.
  logical,       optional, intent(in) :: permitDiminishing !< do not flag error if the
                                                           !! extrema are diminishing.

  ! Local variables
  real, dimension(G%isc:G%iec, G%jsc:G%jec) :: &
    tmp_A, &  ! The area per cell [L2 ~> m2]
    tmp_V, &  ! The column-integrated volume or mass [H L2 ~> m3 or kg],
              ! depending on whether the Boussinesq approximation is used
    tmp_T, &  ! The column-integrated temperature [C H L2 ~> degC m3 or degC kg]
    tmp_S     ! The column-integrated salinity [S H L2 ~> ppt m3 or ppt kg]
  real :: Vol, dV    ! The total ocean volume or mass and its change [H L2 ~> m3 or kg]
  real :: Area       ! The total ocean surface area [L2 ~> m2].
  real :: h_minimum  ! The minimum layer thicknesses [H ~> m or kg m-2]
  real :: T_scale    ! The scaling conversion factor for temperatures [degC C-1 ~> 1]
  real :: S_scale    ! The scaling conversion factor for salinities [ppt S-1 ~> 1]
  logical :: do_TS   ! If true, evaluate statistics for temperature and salinity
  type(stats) :: T, delT ! Temperature statistics in unscaled units [degC]
  type(stats) :: S, delS ! Salinity statistics in unscaled units [ppt]

  ! NOTE: save data is not normally allowed but we use it for debugging purposes here on the
  !       assumption we will not turn this on with threads
  type(stats), save :: oldT, oldS
  logical, save :: firstCall = .true.
  real, save :: oldVol ! The previous total ocean volume or mass [H L2 ~> m3 or kg]

  character(len=80) :: lMsg
  integer :: is, ie, js, je, nz, i, j, k

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke
  do_TS = associated(Temp) .and. associated(Salt)

  tmp_A(:,:) = 0.0
  tmp_V(:,:) = 0.0
  tmp_T(:,:) = 0.0
  tmp_S(:,:) = 0.0

  T_scale = US%C_to_degC ; S_scale = US%S_to_ppt

  ! First collect local stats
  do j=js,je ; do i=is,ie
    tmp_A(i,j) = tmp_A(i,j) + G%areaT(i,j)
  enddo ; enddo
  T%minimum = 1.E34 ; T%maximum = -1.E34 ; T%average = 0.
  S%minimum = 1.E34 ; S%maximum = -1.E34 ; S%average = 0.
  h_minimum = 1.E34*GV%m_to_H
  do k=1,nz ; do j=js,je ; do i=is,ie
    if (G%mask2dT(i,j)>0.) then
      dV = G%areaT(i,j)*h(i,j,k)
      tmp_V(i,j) = tmp_V(i,j) + dV
      if (do_TS .and. h(i,j,k)>0.) then
        T%minimum = min( T%minimum, T_scale*Temp(i,j,k) ) ; T%maximum = max( T%maximum, T_scale*Temp(i,j,k) )
        S%minimum = min( S%minimum, S_scale*Salt(i,j,k) ) ; S%maximum = max( S%maximum, S_scale*Salt(i,j,k) )
        tmp_T(i,j) = tmp_T(i,j) + dV*Temp(i,j,k)
        tmp_S(i,j) = tmp_S(i,j) + dV*Salt(i,j,k)
      endif
      if (h_minimum > h(i,j,k)) h_minimum = h(i,j,k)
    endif
  enddo ; enddo ; enddo
  Area = reproducing_sum( tmp_A, unscale=US%L_to_m**2 )
  Vol = reproducing_sum( tmp_V, unscale=US%L_to_m**2*GV%H_to_mks )
  if (do_TS) then
    call min_across_PEs( T%minimum ) ; call max_across_PEs( T%maximum )
    call min_across_PEs( S%minimum ) ; call max_across_PEs( S%maximum )
    T%average = T_scale*reproducing_sum( tmp_T, unscale=US%C_to_degC*US%L_to_m**2*GV%H_to_mks) / Vol
    S%average = S_scale*reproducing_sum( tmp_S, unscale=US%S_to_ppt*US%L_to_m**2*GV%H_to_mks) / Vol
  endif
  if (is_root_pe()) then
    if (.not.firstCall) then
      dV = Vol - oldVol
      delT%minimum = T%minimum - oldT%minimum ; delT%maximum = T%maximum - oldT%maximum
      delT%average = T%average - oldT%average
      delS%minimum = S%minimum - oldS%minimum ; delS%maximum = S%maximum - oldS%maximum
      delS%average = S%average - oldS%average
      write(lMsg(1:80),'(2(a,es12.4))') 'Mean thickness =', GV%H_to_mks*Vol/Area,' frac. delta=',dV/Vol
      call MOM_mesg(lMsg//trim(mesg))
      if (do_TS) then
        write(lMsg(1:80),'(a,3es12.4)') 'Temp min/mean/max =',T%minimum,T%average,T%maximum
        call MOM_mesg(lMsg//trim(mesg))
        write(lMsg(1:80),'(a,3es12.4)') 'delT min/mean/max =',delT%minimum,delT%average,delT%maximum
        call MOM_mesg(lMsg//trim(mesg))
        write(lMsg(1:80),'(a,3es12.4)') 'Salt min/mean/max =',S%minimum,S%average,S%maximum
        call MOM_mesg(lMsg//trim(mesg))
        write(lMsg(1:80),'(a,3es12.4)') 'delS min/mean/max =',delS%minimum,delS%average,delS%maximum
        call MOM_mesg(lMsg//trim(mesg))
      endif
    else
      write(lMsg(1:80),'(a,es12.4)') 'Mean thickness =', GV%H_to_mks*Vol/Area
      call MOM_mesg(lMsg//trim(mesg))
      if (do_TS) then
        write(lMsg(1:80),'(a,3es12.4)') 'Temp min/mean/max =', T%minimum, T%average, T%maximum
        call MOM_mesg(lMsg//trim(mesg))
        write(lMsg(1:80),'(a,3es12.4)') 'Salt min/mean/max =', S%minimum, S%average, S%maximum
        call MOM_mesg(lMsg//trim(mesg))
      endif
    endif
  endif
  firstCall = .false. ; oldVol = Vol
  oldT%minimum = T%minimum ; oldT%maximum = T%maximum ; oldT%average = T%average
  oldS%minimum = S%minimum ; oldS%maximum = S%maximum ; oldS%average = S%average

  if (do_TS .and. T%minimum<-5.0) then
    do j=js,je ; do i=is,ie
      if (minval(T_scale*Temp(i,j,:)) == T%minimum) then
        write(0,'(a,2f12.5)') 'x,y=', G%geoLonT(i,j), G%geoLatT(i,j)
        write(0,'(a3,3a12)') 'k','h','Temp','Salt'
        do k = 1, nz
          write(0,'(i3,3es12.4)') k, h(i,j,k), T_scale*Temp(i,j,k), S_scale*Salt(i,j,k)
        enddo
        stop 'Extremum detected'
      endif
    enddo ; enddo
  endif

  if (h_minimum<0.0) then
    do j=js,je ; do i=is,ie
      if (minval(h(i,j,:)) == h_minimum) then
        write(0,'(a,2f12.5)') 'x,y=',G%geoLonT(i,j),G%geoLatT(i,j)
        write(0,'(a3,3a12)') 'k','h','Temp','Salt'
        do k = 1, nz
          write(0,'(i3,3es12.4)') k, h(i,j,k), T_scale*Temp(i,j,k), S_scale*Salt(i,j,k)
        enddo
        stop 'Negative thickness detected'
      endif
    enddo ; enddo
  endif

end subroutine MOM_state_stats

end module MOM_checksum_packages
