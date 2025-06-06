!> Configures the models sponges for the Rotating Gravity Current (RGC) experiment.
module RGC_initialization

! This file is part of MOM6. See LICENSE.md for the license.

!***********************************************************************
!* By Elizabeth Yankovsky, May 2018                                    *
!***********************************************************************

use MOM_ALE_sponge, only : ALE_sponge_CS, set_up_ALE_sponge_field, initialize_ALE_sponge
use MOM_ALE_sponge, only : set_up_ALE_sponge_vel_field
use MOM_domains, only : pass_var
use MOM_dyn_horgrid, only : dyn_horgrid_type
use MOM_error_handler, only : MOM_mesg, MOM_error, FATAL, is_root_pe, WARNING
use MOM_file_parser, only : get_param, log_version, param_file_type
use MOM_get_input, only : directories
use MOM_grid, only : ocean_grid_type
use MOM_io, only : file_exists, MOM_read_data, slasher
use MOM_sponge, only : sponge_CS, set_up_sponge_field, initialize_sponge
use MOM_sponge, only : set_up_sponge_ML_density
use MOM_unit_scaling, only : unit_scale_type
use MOM_variables, only : thermo_var_ptrs
use MOM_verticalGrid, only : verticalGrid_type
use MOM_EOS, only : calculate_density, EOS_domain
implicit none ; private

#include <MOM_memory.h>

public RGC_initialize_sponges

! A note on unit descriptions in comments: MOM6 uses units that can be rescaled for dimensional
! consistency testing. These are noted in comments with units like Z, H, L, and T, along with
! their mks counterparts with notation like "a velocity [Z T-1 ~> m s-1]".  If the units
! vary with the Boussinesq approximation, the Boussinesq variant is given first.

contains

!> Sets up the the inverse restoration time, and the values towards which the interface heights,
!! velocities and tracers should be restored within the sponges for the RGC test case.
subroutine RGC_initialize_sponges(G, GV, US, tv, u, v, depth_tot, PF, use_ALE, CSp, ACSp)
  type(ocean_grid_type),   intent(in) :: G  !< The ocean's grid structure.
  type(verticalGrid_type), intent(in) :: GV !< The ocean's vertical grid structure.
  type(unit_scale_type),   intent(in) :: US !< A dimensional unit scaling type
  type(thermo_var_ptrs),   intent(in) :: tv !< A structure containing pointers
                                            !! to any available thermodynamic
                                            !! fields, potential temperature and
                                            !! salinity or mixed layer density.
                                            !! Absent fields have NULL pointers.
  real, dimension(SZIB_(G),SZJ_(G),SZK_(GV)), &
                 target, intent(in) :: u    !< Array with the u velocity [L T-1 ~> m s-1]
  real, dimension(SZI_(G),SZJB_(G),SZK_(GV)), &
                 target, intent(in) :: v    !< Array with the v velocity [L T-1 ~> m s-1]
  real, dimension(SZI_(G),SZJ_(G)), &
                         intent(in) :: depth_tot  !< The nominal total depth of the ocean [Z ~> m]
  type(param_file_type), intent(in) :: PF   !< A structure to parse for model parameter values.
  logical,               intent(in) :: use_ALE !< If true, indicates model is in ALE mode
  type(sponge_CS),       pointer    :: CSp  !< Layer-mode sponge structure
  type(ALE_sponge_CS),   pointer    :: ACSp !< ALE-mode sponge structure

  ! Local variables
  real :: T(SZI_(G),SZJ_(G),SZK_(GV)) ! A temporary array for temperature [C ~> degC]
  real :: S(SZI_(G),SZJ_(G),SZK_(GV)) ! A temporary array for salinity [S ~> ppt]
  real :: U1(SZIB_(G),SZJ_(G),SZK_(GV)) ! A temporary array for u [L T-1 ~> m s-1]
  real :: V1(SZI_(G),SZJB_(G),SZK_(GV)) ! A temporary array for v [L T-1 ~> m s-1]
  real :: rho(SZI_(G),SZJ_(G))      ! A temporary array for mixed layer density [R ~> kg m-3].
  real :: dz(SZI_(G),SZJ_(G),SZK_(GV)) ! Sponge layer thicknesses in height units [Z ~> m]
  real :: Idamp(SZI_(G),SZJ_(G))    ! The sponge damping rate at h points [T-1 ~> s-1]
  real :: TNUDG                     ! Nudging time scale [T ~> s]
  real :: pres(SZI_(G))             ! An array of the reference pressure [R L2 T-2 ~> Pa]
  real :: eta(SZI_(G),SZJ_(G),SZK_(GV)+1) ! A temporary array for eta, positive upward [Z ~> m]
  logical :: sponge_uv              ! Nudge velocities (u and v) towards zero
  real :: min_depth                 ! The minimum depth of the ocean [Z ~> m]
  real :: dummy1                    ! The position relative to the sponge width [nondim]
  real :: min_thickness             ! A minimum layer thickness [H ~> m or kg m-2] (unused)
  real :: lensponge                 ! The width of the sponge in axis units, [km] or [m]
  character(len=40) :: filename, state_file
  character(len=40) :: temp_var, salt_var, eta_var, inputdir, h_var

  character(len=40)  :: mdl = "RGC_initialize_sponges" ! This subroutine's name.
  integer, dimension(2) :: EOSdom ! The i-computational domain for the equation of state
  integer :: i, j, is, ie, js, je, isd, ied, jsd, jed, nz, iscB, iecB, jscB, jecB

  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke
  isd = G%isd ; ied = G%ied ; jsd = G%jsd ; jed = G%jed
  iscB = G%iscB ; iecB = G%iecB; jscB = G%jscB ; jecB = G%jecB

  ! The variable min_thickness is unused, and can probably be eliminated.
  call get_param(PF, mdl, "MIN_THICKNESS", min_thickness, 'Minimum layer thickness', &
                 units='m', default=1.e-3, scale=GV%m_to_H)

  call get_param(PF, mdl, "RGC_TNUDG", TNUDG, 'Nudging time scale for sponge layers', &
                 units='days', default=0.0, scale=86400.0*US%s_to_T)

  call get_param(PF, mdl, "LENSPONGE", lensponge, &
                 "The length of the sponge layer.", &
                 units=G%x_ax_unit_short, default=10.0)

  call get_param(PF, mdl, "SPONGE_UV", sponge_uv, &
                 "Nudge velocities (u and v) towards zero in the sponge layer.", &
                 default=.false., do_not_log=.true.)

  T(:,:,:) = 0.0 ; S(:,:,:) = 0.0 ; Idamp(:,:) = 0.0

  call get_param(PF, mdl, "MINIMUM_DEPTH", min_depth, &
                 "The minimum depth of the ocean.", units="m", default=0.0, scale=US%m_to_Z)

  if (associated(CSp)) call MOM_error(FATAL, &
          "RGC_initialize_sponges called with an associated control structure.")
  if (associated(ACSp)) call MOM_error(FATAL, &
          "RGC_initialize_sponges called with an associated ALE-sponge control structure.")

  !  Here the inverse damping time [T-1 ~> s-1], is set. Set Idamp to 0
  !  wherever there is no sponge, and the subroutines that are called
  !  will automatically set up the sponges only where Idamp is positive
  !  and mask2dT is 1.

  do j=js,je ; do i=is,ie
    if ((depth_tot(i,j) <= min_depth) .or. (G%geoLonT(i,j) <= lensponge)) then
      Idamp(i,j) = 0.0
    elseif (G%geoLonT(i,j) >= (G%len_lon - lensponge) .AND. G%geoLonT(i,j) <= G%len_lon) then
      dummy1 = (G%geoLonT(i,j)-(G%len_lon - lensponge))/(lensponge)
      Idamp(i,j) = (1.0/TNUDG) * max(0.0,dummy1)
    else
      Idamp(i,j) = 0.0
    endif
  enddo ; enddo


  ! 1) Read eta, salt and temp from IC file
  call get_param(PF, mdl, "INPUTDIR", inputdir, default=".")
  inputdir = slasher(inputdir)
  call get_param(PF, mdl, "RGC_SPONGE_FILE", state_file, &
              "The name of the file with temps., salts. and interfaces to \n"// &
              " damp toward.", fail_if_missing=.true.)
  call get_param(PF, mdl, "SPONGE_PTEMP_VAR", temp_var, &
              "The name of the potential temperature variable in \n"//&
              "SPONGE_STATE_FILE.", default="Temp")
  call get_param(PF, mdl, "SPONGE_SALT_VAR", salt_var, &
              "The name of the salinity variable in \n"//&
              "SPONGE_STATE_FILE.", default="Salt")
  call get_param(PF, mdl, "SPONGE_ETA_VAR", eta_var, &
              "The name of the interface height variable in \n"//&
              "SPONGE_STATE_FILE.", default="eta")
  call get_param(PF, mdl, "SPONGE_H_VAR", h_var, &
              "The name of the layer thickness variable in \n"//&
              "SPONGE_STATE_FILE.", default="h")

  !read temp and eta
  filename = trim(inputdir)//trim(state_file)
  if (.not.file_exists(filename, G%Domain)) &
      call MOM_error(FATAL, " RGC_initialize_sponges: Unable to open "//trim(filename))
  call MOM_read_data(filename, temp_var, T(:,:,:), G%Domain, scale=US%degC_to_C)
  call MOM_read_data(filename, salt_var, S(:,:,:), G%Domain, scale=US%ppt_to_S)
  if (use_ALE) then

    call MOM_read_data(filename, h_var, dz(:,:,:), G%Domain, scale=US%m_to_Z)
    call pass_var(dz, G%domain)

    call initialize_ALE_sponge(Idamp, G, GV, PF, ACSp, dz, nz, data_h_is_Z=.true.)

    !  The remaining calls to set_up_sponge_field can be in any order.
    if ( associated(tv%T) ) call set_up_ALE_sponge_field(T, G, GV, tv%T, ACSp, 'temp', &
        sp_long_name='temperature', sp_unit='degC s-1')
    if ( associated(tv%S) ) call set_up_ALE_sponge_field(S, G, GV, tv%S, ACSp, 'salt', &
        sp_long_name='salinity', sp_unit='g kg-1 s-1')

    if (sponge_uv) then
      U1(:,:,:) = 0.0 ; V1(:,:,:) = 0.0
      call set_up_ALE_sponge_vel_field(U1, V1, G, GV, u, v, ACSp)
    endif


  else ! layer mode

    !read eta
    call MOM_read_data(filename, eta_var, eta(:,:,:), G%Domain, scale=US%m_to_Z)

    ! Set the sponge damping rates so that the model will know where to
    ! apply the sponges, along with the interface heights.
    call initialize_sponge(Idamp, eta, G, PF, CSp, GV)

    if ( GV%nkml>0 ) then
    !   This call to set_up_sponge_ML_density registers the target values of the
    ! mixed layer density, which is used in determining which layers can be
    ! inflated without causing static instabilities.
      do i=is,ie ; pres(i) = tv%P_Ref ; enddo
      EOSdom(:) = EOS_domain(G%HI)
      do j=js,je
        call calculate_density(T(:,j,1), S(:,j,1), pres, rho(:,j), tv%eqn_of_state, EOSdom)
      enddo

      call set_up_sponge_ML_density(rho, G, CSp)
    endif

    ! Apply sponge in tracer fields
    call set_up_sponge_field(T, tv%T, G, GV, nz, CSp)
    call set_up_sponge_field(S, tv%S, G, GV, nz, CSp)

  endif

end subroutine RGC_initialize_sponges

end module RGC_initialization
