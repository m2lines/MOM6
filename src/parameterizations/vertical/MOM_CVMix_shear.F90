!> Interface to CVMix interior shear schemes
module MOM_CVMix_shear

! This file is part of MOM6. See LICENSE.md for the license.

!> \author Brandon Reichl

use MOM_diag_mediator, only : post_data, register_diag_field, safe_alloc_ptr
use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_error_handler, only : MOM_error, is_root_pe, FATAL, WARNING, NOTE
use MOM_file_parser, only : get_param, log_version, param_file_type
use MOM_grid, only : ocean_grid_type
use MOM_interface_heights, only : thickness_to_dz
use MOM_unit_scaling, only : unit_scale_type
use MOM_variables, only : thermo_var_ptrs
use MOM_verticalGrid, only : verticalGrid_type
use MOM_EOS, only : calculate_density
use CVMix_shear, only : CVMix_init_shear, CVMix_coeffs_shear
use MOM_kappa_shear, only : kappa_shear_is_used
implicit none ; private

#include <MOM_memory.h>

public calculate_CVMix_shear, CVMix_shear_init, CVMix_shear_is_used, CVMix_shear_end

! A note on unit descriptions in comments: MOM6 uses units that can be rescaled for dimensional
! consistency testing. These are noted in comments with units like Z, H, L, and T, along with
! their mks counterparts with notation like "a velocity [Z T-1 ~> m s-1]".  If the units
! vary with the Boussinesq approximation, the Boussinesq variant is given first.

!> Control structure including parameters for CVMix interior shear schemes.
type, public :: CVMix_shear_cs ; private
  logical :: use_LMD94                      !< Flags to use the LMD94 scheme
  logical :: use_PP81                       !< Flags to use Pacanowski and Philander (JPO 1981)
  integer :: n_smooth_ri                    !< Number of times to smooth Ri using a 1-2-1 filter
  real    :: Ri_zero                        !< LMD94 critical Richardson number [nondim]
  real    :: Nu_zero                        !< LMD94 maximum interior diffusivity [Z2 T-1 ~> m2 s-1]
  real    :: KPP_exp                        !< Exponent of unitless factor of diffusivities
                                            !! for KPP internal shear mixing scheme [nondim]
  real, allocatable, dimension(:,:,:) :: N2 !< Squared Brunt-Vaisala frequency [T-2 ~> s-2]
  real, allocatable, dimension(:,:,:) :: S2 !< Squared shear frequency [T-2 ~> s-2]
  real, allocatable, dimension(:,:,:) :: ri_grad !< Gradient Richardson number [nondim]
  real, allocatable, dimension(:,:,:) :: ri_grad_orig !< Gradient Richardson number
                                                      !! after smoothing [nondim]
  character(10) :: Mix_Scheme               !< Mixing scheme name (string)

  type(diag_ctrl), pointer :: diag => NULL() !< Pointer to the diagnostics control structure
  !>@{ Diagnostic handles
  integer :: id_N2 = -1, id_S2 = -1, id_ri_grad = -1, id_kv = -1, id_kd = -1
  integer :: id_ri_grad_orig = -1
  !>@}

end type CVMix_shear_cs

character(len=40)  :: mdl = "MOM_CVMix_shear"  !< This module's name.

contains

!> Subroutine for calculating (internal) vertical diffusivities/viscosities
subroutine calculate_CVMix_shear(u_H, v_H, h, tv, kd, kv, G, GV, US, CS )
  type(ocean_grid_type),                      intent(in)  :: G   !< Grid structure.
  type(verticalGrid_type),                    intent(in)  :: GV  !< Vertical grid structure.
  type(unit_scale_type),                      intent(in)  :: US  !< A dimensional unit scaling type
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)  :: u_H !< Initial zonal velocity on T points [L T-1 ~> m s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)  :: v_H !< Initial meridional velocity on T
                                                                 !! points [L T-1 ~> m s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)),  intent(in)  :: h   !< Layer thickness [H ~> m or kg m-2].
  type(thermo_var_ptrs),                      intent(in)  :: tv  !< Thermodynamics structure.
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1), intent(out) :: kd !< The vertical diffusivity at each interface
                                                                 !! (not layer!) [H Z T-1 ~> m2 s-1 or kg m-1 s-1]
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)+1), intent(out) :: kv !< The vertical viscosity at each interface
                                                                 !! (not layer!) [H Z T-1 ~> m2 s-1 or Pa s]
  type(CVMix_shear_cs),                       pointer     :: CS  !< The control structure returned by a previous
                                                                 !! call to CVMix_shear_init.
  ! Local variables
  integer :: i, j, k, kk, km1, s
  real :: GoRho  ! Gravitational acceleration divided by density [Z T-2 R-1 ~> m4 s-2 kg-1]
  real :: pref   ! Interface pressures [R L2 T-2 ~> Pa]
  real :: DU, DV ! Velocity differences [L T-1 ~> m s-1]
  real :: dz_int ! Grid spacing around an interface [Z ~> m]
  real :: N2     ! Buoyancy frequency at an interface [T-2 ~> s-2]
  real :: S2     ! Shear squared at an interface [T-2 ~> s-2]
  real :: dummy  ! A dummy variable [nondim]
  real :: dRho   ! Buoyancy differences [Z T-2 ~> m s-2]
  real, dimension(SZI_(G),SZK_(GV)) :: dz ! Height change across layers [Z ~> m]
  real, dimension(2*(GV%ke)) :: pres_1d ! A column of interface pressures [R L2 T-2 ~> Pa]
  real, dimension(2*(GV%ke)) :: temp_1d ! A column of temperatures [C ~> degC]
  real, dimension(2*(GV%ke)) :: salt_1d ! A column of salinities [S ~> ppt]
  real, dimension(2*(GV%ke)) :: rho_1d  ! A column of densities at interface pressures [R ~> kg m-3]
  real, dimension(GV%ke+1) :: Ri_Grad   !< Gradient Richardson number [nondim]
  real, dimension(GV%ke+1) :: Ri_Grad_prev !< Gradient Richardson number before s.th smoothing iteration [nondim]
  real, dimension(GV%ke+1) :: Kvisc   !< Vertical viscosity at interfaces [m2 s-1]
  real, dimension(GV%ke+1) :: Kdiff   !< Diapycnal diffusivity at interfaces [m2 s-1]
  real :: epsln  !< Threshold to identify vanished layers [H ~> m or kg m-2]

  ! some constants
  GoRho = GV%g_Earth_Z_T2 / GV%Rho0
  epsln = 1.e-10 * GV%m_to_H

  do j = G%jsc, G%jec

    ! Find the vertical distances across layers.
    call thickness_to_dz(h, tv, dz, j, G, GV)

    do i = G%isc, G%iec

      ! skip calling for land points
      if (G%mask2dT(i,j)==0.) cycle

      ! Richardson number computed for each cell in a column.
      pRef = 0. ; if (associated(tv%p_surf)) pRef = tv%p_surf(i,j)
      Ri_Grad(:)=1.e8 !Initialize w/ large Richardson value
      do k=1,GV%ke
        ! pressure, temp, and saln for EOS
        ! kk+1 = k fields
        ! kk+2 = km1 fields
        km1  = max(1, k-1)
        kk   = 2*(k-1)
        pres_1D(kk+1) = pRef
        pres_1D(kk+2) = pRef
        Temp_1D(kk+1) = tv%T(i,j,k)
        Temp_1D(kk+2) = tv%T(i,j,km1)
        Salt_1D(kk+1) = tv%S(i,j,k)
        Salt_1D(kk+2) = tv%S(i,j,km1)

        ! pRef is pressure at interface between k and km1.
        ! iterate pRef for next pass through k-loop.
        pRef = pRef + (GV%g_Earth * GV%H_to_RZ) * h(i,j,k)

      enddo ! k-loop finishes

      ! compute in-situ density [R ~> kg m-3]
      call calculate_density(Temp_1D, Salt_1D, pres_1D, rho_1D, tv%eqn_of_state)

      ! N2 (can be negative) on interface
      do k = 1, GV%ke
        km1 = max(1, k-1)
        kk = 2*(k-1)
        DU = u_h(i,j,k) - u_h(i,j,km1)
        DV = v_h(i,j,k) - v_h(i,j,km1)
        if (GV%Boussinesq .or. GV%semi_Boussinesq) then
          dRho = GoRho * (rho_1D(kk+1) - rho_1D(kk+2))
        else
          dRho = GV%g_Earth_Z_T2 * (rho_1D(kk+1) - rho_1D(kk+2)) / (0.5*(rho_1D(kk+1) + rho_1D(kk+2)))
        endif
        dz_int = 0.5*(dz(i,km1) + dz(i,k)) + GV%dZ_subroundoff
        N2 = DRHO / dz_int
        S2 = US%L_to_Z**2*((DU*DU) + (DV*DV)) / (dz_int*dz_int)
        Ri_Grad(k) = max(0., N2) / max(S2, 1.e-10*US%T_to_s**2)

        ! fill 3d arrays, if user asks for diagnostics
        if (CS%id_N2 > 0) CS%N2(i,j,k) = N2
        if (CS%id_S2 > 0) CS%S2(i,j,k) = S2

      enddo

      Ri_grad(GV%ke+1) = Ri_grad(GV%ke)

      if (CS%n_smooth_ri > 0) then

        if (CS%id_ri_grad_orig > 0) CS%ri_grad_orig(i,j,:) = Ri_Grad(:)

        ! 1) fill Ri_grad in vanished layers with adjacent value
        do k = 2, GV%ke
          if (h(i,j,k) <= epsln) Ri_grad(k) = Ri_grad(k-1)
        enddo

        Ri_grad(GV%ke+1) = Ri_grad(GV%ke)

        do s=1,CS%n_smooth_ri

          Ri_Grad_prev(:) = Ri_Grad(:)

          ! 2) vertically smooth Ri with 1-2-1 filter
          dummy =  0.25 * Ri_grad_prev(2)
          do k = 3, GV%ke
            Ri_Grad(k) = dummy + 0.5 * Ri_Grad_prev(k) + 0.25 * Ri_grad_prev(k+1)
            dummy = 0.25 * Ri_grad(k)
          enddo
        enddo

        Ri_grad(GV%ke+1) = Ri_grad(GV%ke)

      endif

      if (CS%id_ri_grad > 0) CS%ri_grad(i,j,:) = Ri_Grad(:)

      do K=1,GV%ke+1
        Kvisc(K) = GV%HZ_T_to_m2_s * kv(i,j,K)
        Kdiff(K) = GV%HZ_T_to_m2_s * kd(i,j,K)
      enddo

      ! Call to CVMix wrapper for computing interior mixing coefficients.
      call  CVMix_coeffs_shear(Mdiff_out=Kvisc(:), &
                                   Tdiff_out=Kdiff(:), &
                                   RICH=Ri_Grad(:), &
                                   nlev=GV%ke,    &
                                   max_nlev=GV%ke)
      do K=1,GV%ke+1
        kv(i,j,K) = GV%m2_s_to_HZ_T * Kvisc(K)
        kd(i,j,K) = GV%m2_s_to_HZ_T * Kdiff(K)
      enddo
    enddo
  enddo

  ! write diagnostics
  if (CS%id_kd > 0) call post_data(CS%id_kd, kd, CS%diag)
  if (CS%id_kv > 0) call post_data(CS%id_kv, kv, CS%diag)
  if (CS%id_N2 > 0) call post_data(CS%id_N2, CS%N2, CS%diag)
  if (CS%id_S2 > 0) call post_data(CS%id_S2, CS%S2, CS%diag)
  if (CS%id_ri_grad > 0) call post_data(CS%id_ri_grad, CS%ri_grad, CS%diag)
  if (CS%id_ri_grad_orig > 0) call post_data(CS%id_ri_grad_orig ,CS%ri_grad_orig, CS%diag)

end subroutine calculate_CVMix_shear


!> Initialized the CVMix internal shear mixing routine.
!! \todo Does this note require emphasis?
!! \note *This is where we test to make sure multiple internal shear
!!       mixing routines (including JHL) are not enabled at the same time.*
!! (returns) CVMix_shear_init - True if module is to be used, False otherwise
logical function CVMix_shear_init(Time, G, GV, US, param_file, diag, CS)
  type(time_type),         intent(in)    :: Time !< The current time.
  type(ocean_grid_type),   intent(in)    :: G  !< Grid structure.
  type(verticalGrid_type), intent(in)    :: GV !< Vertical grid structure.
  type(unit_scale_type),   intent(in)    :: US !< A dimensional unit scaling type
  type(param_file_type),   intent(in)    :: param_file !< Run-time parameter file handle
  type(diag_ctrl), target, intent(inout) :: diag !< Diagnostics control structure.
  type(CVMix_shear_cs),    pointer       :: CS !< This module's control structure.
  ! Local variables
  integer :: NumberTrue=0
  logical :: use_JHL
  logical :: use_LMD94
  logical :: use_PP81

! This include declares and sets the variable "version".
#include "version_variable.h"

  if (associated(CS)) then
    call MOM_error(WARNING, "CVMix_shear_init called with an associated "// &
                            "control structure.")
    return
  endif

! Set default, read and log parameters
  call get_param(param_file, mdl, "USE_LMD94", use_LMD94, default=.false., do_not_log=.true.)
  call get_param(param_file, mdl, "USE_PP81", use_PP81, default=.false., do_not_log=.true.)
  call log_version(param_file, mdl, version, &
           "Parameterization of shear-driven turbulence via CVMix (various options)", &
            all_default=.not.(use_PP81.or.use_LMD94))
  call get_param(param_file, mdl, "USE_LMD94", use_LMD94, &
                 "If true, use the Large-McWilliams-Doney (JGR 1994) "//&
                 "shear mixing parameterization.", default=.false.)
  if (use_LMD94) &
    NumberTrue=NumberTrue + 1
  call get_param(param_file, mdl, "USE_PP81", use_PP81, &
                 "If true, use the Pacanowski and Philander (JPO 1981) "//&
                 "shear mixing parameterization.", default=.false.)
  if (use_PP81) &
    NumberTrue = NumberTrue + 1
  use_JHL=kappa_shear_is_used(param_file)
  if (use_JHL) NumberTrue = NumberTrue + 1
  ! After testing for interior schemes, make sure only 0 or 1 are enabled.
  ! Otherwise, warn user and kill job.
  if ((NumberTrue) > 1) then
    call MOM_error(FATAL, 'MOM_CVMix_shear_init: '// &
           'Multiple shear driven internal mixing schemes selected,'//&
           ' please disable all but one scheme to proceed.')
  endif

  CVMix_shear_init = use_PP81 .or. use_LMD94

  ! Forego remainder of initialization if not using this scheme
  if (.not. CVMix_shear_init) return

  allocate(CS)
  CS%use_LMD94 = use_LMD94
  CS%use_PP81 = use_PP81
  if (use_LMD94) &
    CS%Mix_Scheme = 'KPP'
  if (use_PP81) &
    CS%Mix_Scheme = 'PP'

  call get_param(param_file, mdl, "NU_ZERO", CS%Nu_Zero, &
                 "Leading coefficient in KPP shear mixing.", &
                 units="m2 s-1", default=5.e-3, scale=US%m2_s_to_Z2_T)
  call get_param(param_file, mdl, "RI_ZERO", CS%Ri_Zero, &
                 "Critical Richardson for KPP shear mixing, "// &
                 "NOTE this the internal mixing and this is "// &
                 "not for setting the boundary layer depth.", &
                 units="nondim", default=0.8)
  call get_param(param_file, mdl, "KPP_EXP", CS%KPP_exp, &
                 "Exponent of unitless factor of diffusivities, "// &
                 "for KPP internal shear mixing scheme.", &
                 units="nondim", default=3.0)
  call get_param(param_file, mdl, "N_SMOOTH_RI", CS%n_smooth_ri, &
                 "If > 0, vertically smooth the Richardson "// &
                 "number by applying a 1-2-1 filter N_SMOOTH_RI times.", &
                 default=0)
  call cvmix_init_shear(mix_scheme=CS%Mix_Scheme, &
                        KPP_nu_zero=US%Z2_T_to_m2_s*CS%Nu_Zero,   &
                        KPP_Ri_zero=CS%Ri_zero,   &
                        KPP_exp=CS%KPP_exp)

  ! Register diagnostics; allocation and initialization
  CS%diag => diag

  CS%id_N2 = register_diag_field('ocean_model', 'N2_shear', diag%axesTi, Time, &
      'Square of Brunt-Vaisala frequency used by MOM_CVMix_shear module', '1/s2', conversion=US%s_to_T**2)
  if (CS%id_N2 > 0) then
    allocate( CS%N2( SZI_(G), SZJ_(G), SZK_(GV)+1 ), source=0. )
  endif

  CS%id_S2 = register_diag_field('ocean_model', 'S2_shear', diag%axesTi, Time, &
      'Square of vertical shear used by MOM_CVMix_shear module','1/s2', conversion=US%s_to_T**2)
  if (CS%id_S2 > 0) then
    allocate( CS%S2( SZI_(G), SZJ_(G), SZK_(GV)+1 ), source=0. )
  endif

  CS%id_ri_grad = register_diag_field('ocean_model', 'ri_grad_shear', diag%axesTi, Time, &
      'Gradient Richarson number used by MOM_CVMix_shear module','nondim')
  if (CS%id_ri_grad > 0) then !Initialize w/ large Richardson value
    allocate( CS%ri_grad( SZI_(G), SZJ_(G), SZK_(GV)+1 ), source=1.e8 )
  endif

  if (CS%n_smooth_ri > 0) then
    CS%id_ri_grad_orig = register_diag_field('ocean_model', 'ri_grad_shear_orig', &
         diag%axesTi, Time, &
        'Original gradient Richarson number, before smoothing was applied. This is '//&
        'part of the MOM_CVMix_shear module and only available when N_SMOOTH_RI > 0','nondim')
  endif
  if (CS%id_ri_grad_orig > 0 .or. CS%n_smooth_ri > 0) then !Initialize w/ large Richardson value
    allocate( CS%ri_grad_orig( SZI_(G), SZJ_(G), SZK_(GV)+1 ), source=1.e8 )
  endif

  CS%id_kd = register_diag_field('ocean_model', 'kd_shear_CVMix', diag%axesTi, Time, &
      'Vertical diffusivity added by MOM_CVMix_shear module', 'm2/s', conversion=GV%HZ_T_to_m2_s)
  CS%id_kv = register_diag_field('ocean_model', 'kv_shear_CVMix', diag%axesTi, Time, &
      'Vertical viscosity added by MOM_CVMix_shear module', 'm2/s', conversion=GV%HZ_T_to_m2_s)

end function CVMix_shear_init

!> Reads the parameters "USE_LMD94" and "USE_PP81" and returns true if either is true.
!!   This function allows other modules to know whether this parameterization will
!! be used without needing to duplicate the log entry.
logical function CVMix_shear_is_used(param_file)
  type(param_file_type), intent(in) :: param_file !< Run-time parameter files handle.
  ! Local variables
  logical :: LMD94, PP81
  call get_param(param_file, mdl, "USE_LMD94", LMD94, &
                 default=.false., do_not_log=.true.)
  call get_param(param_file, mdl, "USE_PP81", PP81, &
                 default=.false., do_not_log=.true.)
  CVMix_shear_is_used = (LMD94 .or. PP81)
end function CVMix_shear_is_used

!> Clear pointers and deallocate memory
subroutine CVMix_shear_end(CS)
  type(CVMix_shear_cs), intent(inout) :: CS !< Control structure for this module that
                                            !! will be deallocated in this subroutine
  if (CS%id_N2 > 0) deallocate(CS%N2)
  if (CS%id_S2 > 0) deallocate(CS%S2)
  if (CS%id_ri_grad > 0) deallocate(CS%ri_grad)
end subroutine CVMix_shear_end

end module MOM_CVMix_shear
