!> This tracer package dyes flow through open boundaries
module dyed_obc_tracer

! This file is part of MOM6. See LICENSE.md for the license.

use MOM_coupler_types,      only : atmos_ocn_coupler_flux
use MOM_diag_mediator,      only : diag_ctrl
use MOM_error_handler,      only : MOM_error, FATAL, WARNING
use MOM_file_parser,        only : get_param, log_param, log_version, param_file_type
use MOM_forcing_type,       only : forcing
use MOM_hor_index,          only : hor_index_type
use MOM_grid,               only : ocean_grid_type
use MOM_io,                 only : file_exists, MOM_read_data, slasher, vardesc, var_desc, query_vardesc
use MOM_open_boundary,      only : ocean_OBC_type
use MOM_restart,            only : MOM_restart_CS
use MOM_restart,            only : query_initialized, set_initialized
use MOM_time_manager,       only : time_type
use MOM_tracer_registry,    only : register_tracer, tracer_registry_type
use MOM_tracer_diabatic,    only : tracer_vertdiff, applyTracerBoundaryFluxesInOut
use MOM_unit_scaling,       only : unit_scale_type
use MOM_variables,          only : surface
use MOM_verticalGrid,       only : verticalGrid_type
use MOM_open_boundary,      only : OBC_segment_type, register_segment_tracer
use MOM_tracer_registry,    only : tracer_type
use MOM_tracer_registry,    only : tracer_name_lookup
use MOM_tracer_advect_schemes, only : set_tracer_advect_scheme, TracerAdvectionSchemeDoc

implicit none ; private

#include <MOM_memory.h>

public register_dyed_obc_tracer, initialize_dyed_obc_tracer
public dyed_obc_tracer_column_physics, dyed_obc_tracer_end

!> The control structure for the dyed_obc tracer package
type, public :: dyed_obc_tracer_CS ; private
  integer :: ntr    !< The number of tracers that are actually used.
  logical :: coupled_tracers = .false. !< These tracers are not offered to the coupler.
  character(len=200) :: tracer_IC_file !< The full path to the IC file, or " " to initialize internally.
  type(time_type), pointer :: Time => NULL() !< A pointer to the ocean model's clock.
  type(tracer_registry_type), pointer :: tr_Reg => NULL() !< A pointer to the tracer registry
  real, pointer :: tr(:,:,:,:) => NULL()   !< The array of tracers used in this subroutine in [conc]

  logical :: tracers_may_reinit  !< If true, these tracers be set up via the initialization code if
                                 !! they are not found in the restart files.

  integer, allocatable, dimension(:) :: ind_tr !< Indices returned by atmos_ocn_coupler_flux if it is used and the
                                               !! surface tracer concentrations are to be provided to the coupler.

  type(diag_ctrl), pointer :: diag => NULL() !< A structure that is used to
                                   !! regulate the timing of diagnostic output.
  type(MOM_restart_CS), pointer :: restart_CSp => NULL() !< A pointer to the restart control structure

  type(vardesc), allocatable :: tr_desc(:) !< Descriptions and metadata for the tracers
end type dyed_obc_tracer_CS

contains

!> Register tracer fields and subroutines to be used with MOM.
function register_dyed_obc_tracer(HI, GV, param_file, CS, tr_Reg, restart_CS)
  type(hor_index_type),       intent(in) :: HI   !< A horizontal index type structure.
  type(verticalGrid_type),    intent(in) :: GV   !< The ocean's vertical grid structure
  type(param_file_type),      intent(in) :: param_file !< A structure to parse for run-time parameters
  type(dyed_obc_tracer_CS),   pointer    :: CS   !< A pointer that is set to point to the
                                                 !! control structure for this module
  type(tracer_registry_type), pointer    :: tr_Reg !< A pointer to the tracer registry.
  type(MOM_restart_CS), target, intent(inout) :: restart_CS !< MOM restart control struct

  ! Local variables
  character(len=80)  :: name, longname
  ! This include declares and sets the variable "version".
# include "version_variable.h"
  character(len=40)  :: mdl = "dyed_obc_tracer" ! This module's name.
  character(len=200) :: inputdir
  character(len=48)  :: flux_units ! The units for tracer fluxes, usually
                            ! kg(tracer) kg(water)-1 m3 s-1 or kg(tracer) s-1.
  real, pointer :: tr_ptr(:,:,:) => NULL() ! The tracer concentration [conc]
  logical :: register_dyed_obc_tracer
  integer :: isd, ied, jsd, jed, nz, m
  integer :: n_dye           ! Number of regionsl dye tracers
  integer :: advect_scheme   ! Advection scheme value for this tracer
  character(len=256) :: mesg ! Advection scheme name for this tracer

  isd = HI%isd ; ied = HI%ied ; jsd = HI%jsd ; jed = HI%jed ; nz = GV%ke

  if (associated(CS)) then
    call MOM_error(FATAL, "dyed_obc_register_tracer called with an "// &
                          "associated control structure.")
  endif
  allocate(CS)

  ! Read all relevant parameters and write them to the model log.
  call log_version(param_file, mdl, version, "")
  call get_param(param_file, mdl, "NUM_DYED_TRACERS", CS%ntr, &
                 "The number of dyed_obc tracers in this run. Each tracer "//&
                 "should have a separate boundary segment."//&
                 "If not present, use NUM_DYE_TRACERS.", default=-1)
  if (CS%ntr == -1) then
    !for backward compatibility
    call get_param(param_file, mdl, "NUM_DYE_TRACERS", CS%ntr, &
                   "The number of dye tracers in this run. Each tracer "//&
                   "should have a separate boundary segment.", default=0)
    n_dye = 0
  else
    call get_param(param_file, mdl, "NUM_DYE_TRACERS", n_dye, &
                   "The number of dye tracers in this run. Each tracer "//&
                   "should have a separate region.", default=0, do_not_log=.true.)
  endif
  allocate(CS%ind_tr(CS%ntr))
  allocate(CS%tr_desc(CS%ntr))

  call get_param(param_file, mdl, "dyed_obc_TRACER_IC_FILE", CS%tracer_IC_file, &
                 "The name of a file from which to read the initial "//&
                 "conditions for the dyed_obc tracers, or blank to initialize "//&
                 "them internally.", default=" ")
  if (len_trim(CS%tracer_IC_file) >= 1) then
    call get_param(param_file, mdl, "INPUTDIR", inputdir, default=".")
    inputdir = slasher(inputdir)
    CS%tracer_IC_file = trim(inputdir)//trim(CS%tracer_IC_file)
    call log_param(param_file, mdl, "INPUTDIR/dyed_obc_TRACER_IC_FILE", &
                   CS%tracer_IC_file)
  endif

  call get_param(param_file, mdl, "TRACERS_MAY_REINIT", CS%tracers_may_reinit, &
                 "If true, tracers may go through the initialization code "//&
                 "if they are not found in the restart files.  Otherwise "//&
                 "it is a fatal error if the tracers are not found in the "//&
                 "restart files of a restarted run.", default=.false.)

  call get_param(param_file, mdl, "DYED_TRACER_ADVECTION_SCHEME", mesg, &
        desc="The horizontal transport scheme for dyed_obc tracers:\n"//&
        trim(TracerAdvectionSchemeDoc)//&
        "\n Set to blank (the default) to use TRACER_ADVECTION_SCHEME.", default="")

  allocate(CS%tr(isd:ied,jsd:jed,nz,CS%ntr), source=0.0)

  do m=1,CS%ntr
    write(name,'("dye_",I2.2)') m+n_dye  !after regional dye tracers
    write(longname,'("Concentration of dyed_obc Tracer ",I2.2)') m
    CS%tr_desc(m) = var_desc(name, units="kg kg-1", longname=longname, caller=mdl)
    if (GV%Boussinesq) then ; flux_units = "kg kg-1 m3 s-1"
    else ; flux_units = "kg s-1" ; endif

    ! This is needed to force the compiler not to do a copy in the registration
    ! calls.  Curses on the designers and implementers of Fortran90.
    tr_ptr => CS%tr(:,:,:,m)
    ! Get the integer value of the tracer scheme
    call set_tracer_advect_scheme(advect_scheme, mesg)
    ! Register the tracer for horizontal advection, diffusion, and restarts.
    call register_tracer(tr_ptr, tr_Reg, param_file, HI, GV, &
                         name=name, longname=longname, units="kg kg-1", &
                         registry_diags=.true., flux_units=flux_units, &
                         restart_CS=restart_CS, mandatory=.not.CS%tracers_may_reinit, &
                         advect_scheme=advect_scheme)

    !   Set coupled_tracers to be true (hard-coded above) to provide the surface
    ! values to the coupler (if any).  This is meta-code and its arguments will
    ! currently (deliberately) give fatal errors if it is used.
    if (CS%coupled_tracers) &
      CS%ind_tr(m) = atmos_ocn_coupler_flux(trim(name)//'_flux', &
          flux_type=' ', implementation=' ', caller="register_dyed_obc_tracer")
  enddo

  CS%tr_Reg => tr_Reg
  CS%restart_CSp => restart_CS
  register_dyed_obc_tracer = .true.
end function register_dyed_obc_tracer

!> Initializes the CS%ntr tracer fields in tr(:,:,:,:) and sets up the tracer output.
subroutine initialize_dyed_obc_tracer(restart, day, G, GV, h, diag, OBC, CS)
  type(ocean_grid_type),                 intent(in) :: G    !< The ocean's grid structure
  type(verticalGrid_type),               intent(in) :: GV   !< The ocean's vertical grid structure
  logical,                               intent(in) :: restart !< .true. if the fields have already
                                                               !! been read from a restart file.
  type(time_type), target,               intent(in) :: day     !< Time of the start of the run.
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), intent(in) :: h    !< Layer thicknesses [H ~> m or kg m-2]
  type(diag_ctrl), target,               intent(in) :: diag    !< Structure used to regulate diagnostic output.
  type(ocean_OBC_type),                  pointer    :: OBC     !< Structure specifying open boundary options.
  type(dyed_obc_tracer_CS),              pointer    :: CS      !< The control structure returned by a previous
                                                               !! call to dyed_obc_register_tracer.

! Local variables
  character(len=24) :: name     ! A variable's name in a NetCDF file.
  real :: h_neglect         ! A thickness that is so small it is usually lost
                            ! in roundoff and can be neglected [H ~> m or kg m-2].
  integer :: i, j, k, is, ie, js, je, isd, ied, jsd, jed, nz, m
  integer :: IsdB, IedB, JsdB, JedB

  if (.not.associated(CS)) return
  if (CS%ntr < 1) return
  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke
  isd = G%isd ; ied = G%ied ; jsd = G%jsd ; jed = G%jed
  IsdB = G%IsdB ; IedB = G%IedB ; JsdB = G%JsdB ; JedB = G%JedB
  h_neglect = GV%H_subroundoff

  CS%Time => day
  CS%diag => diag

  do m=1,CS%ntr
    if ((.not.restart) .or. (CS%tracers_may_reinit .and. .not. &
        query_initialized(CS%tr(:,:,:,m), name, CS%restart_CSp))) then
      if (len_trim(CS%tracer_IC_file) >= 1) then
        !  Read the tracer concentrations from a netcdf file.
        if (.not.file_exists(CS%tracer_IC_file, G%Domain)) &
          call MOM_error(FATAL, "dyed_obc_initialize_tracer: Unable to open "// &
                          CS%tracer_IC_file)
        call query_vardesc(CS%tr_desc(m), name, caller="initialize_dyed_obc_tracer")
        call MOM_read_data(CS%tracer_IC_file, trim(name), CS%tr(:,:,:,m), G%Domain)
      else
        do k=1,nz ; do j=js,je ; do i=is,ie
          CS%tr(i,j,k,m) = 0.0
        enddo ; enddo ; enddo
      endif
      call set_initialized(CS%tr(:,:,:,m), name, CS%restart_CSp)
    endif ! restart
  enddo ! Tracer loop

end subroutine initialize_dyed_obc_tracer

!> This subroutine applies diapycnal diffusion and any other column
!! tracer physics or chemistry to the tracers from this file.
!! This is a simple example of a set of advected passive tracers.
!!
!! The arguments to this subroutine are redundant in that
!!     h_new(k) = h_old(k) + ea(k) - eb(k-1) + eb(k) - ea(k+1)
subroutine dyed_obc_tracer_column_physics(h_old, h_new,  ea,  eb, fluxes, dt, G, GV, US, CS, &
              evap_CFL_limit, minimum_forcing_depth)
  type(ocean_grid_type),   intent(in) :: G    !< The ocean's grid structure
  type(verticalGrid_type), intent(in) :: GV   !< The ocean's vertical grid structure
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                           intent(in) :: h_old !< Layer thickness before entrainment [H ~> m or kg m-2].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                           intent(in) :: h_new !< Layer thickness after entrainment [H ~> m or kg m-2].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                           intent(in) :: ea   !< an array to which the amount of fluid entrained
                                              !! from the layer above during this call will be
                                              !! added [H ~> m or kg m-2].
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)), &
                           intent(in) :: eb   !< an array to which the amount of fluid entrained
                                              !! from the layer below during this call will be
                                              !! added [H ~> m or kg m-2].
  type(forcing),           intent(in) :: fluxes !< A structure containing pointers to thermodynamic
                                              !! and tracer forcing fields.  Unused fields have NULL ptrs.
  real,                    intent(in) :: dt   !< The amount of time covered by this call [T ~> s]
  type(unit_scale_type),   intent(in) :: US   !< A dimensional unit scaling type
  type(dyed_obc_tracer_CS), pointer   :: CS   !< The control structure returned by a previous
                                              !! call to dyed_obc_register_tracer.
  real,          optional, intent(in) :: evap_CFL_limit !< Limit on the fraction of the water that can
                                              !! be fluxed out of the top layer in a timestep [nondim]
  real,          optional, intent(in) :: minimum_forcing_depth !< The smallest depth over which
                                              !! fluxes can be applied [H ~> m or kg m-2]

! Local variables
  real, dimension(SZI_(G),SZJ_(G),SZK_(GV)) :: h_work ! Used so that h can be modified [H ~> m or kg m-2]
  integer :: i, j, k, is, ie, js, je, nz, m
  is = G%isc ; ie = G%iec ; js = G%jsc ; je = G%jec ; nz = GV%ke

  if (.not.associated(CS)) return
  if (CS%ntr < 1) return

  if (present(evap_CFL_limit) .and. present(minimum_forcing_depth)) then
    do m=1,CS%ntr
      do k=1,nz ;do j=js,je ; do i=is,ie
        h_work(i,j,k) = h_old(i,j,k)
      enddo ; enddo ; enddo
      call applyTracerBoundaryFluxesInOut(G, GV, CS%tr(:,:,:,m), dt, fluxes, h_work, &
                                          evap_CFL_limit, minimum_forcing_depth)
      if (nz > 1) call tracer_vertdiff(h_work, ea, eb, dt, CS%tr(:,:,:,m), G, GV)
    enddo
  else
    do m=1,CS%ntr
      if (nz > 1) call tracer_vertdiff(h_old, ea, eb, dt, CS%tr(:,:,:,m), G, GV)
    enddo
  endif

end subroutine dyed_obc_tracer_column_physics

!> Clean up memory allocations, if any.
subroutine dyed_obc_tracer_end(CS)
  type(dyed_obc_tracer_CS), pointer :: CS   !< The control structure returned by a previous
                                            !! call to dyed_obc_register_tracer.

  if (associated(CS)) then
    if (associated(CS%tr)) deallocate(CS%tr)

    deallocate(CS)
  endif
end subroutine dyed_obc_tracer_end

!> \namespace dyed_obc_tracer
!!
!! By Kate Hedstrom, 2017, copied from DOME tracers and also
!! dye_example.
!!
!! This file contains an example of the code that is needed to set
!! up and use a set of dynamically passive tracers. These tracers
!! dye the inflowing water, one per open boundary segment.
!!
!! A single subroutine is called from within each file to register
!! each of the tracers for reinitialization and advection and to
!! register the subroutine that initializes the tracers and set up
!! their output and the subroutine that does any tracer physics or
!! chemistry along with diapycnal mixing (included here because some
!! tracers may float or swim vertically or dye diapycnal processes).
!!
!! The advection scheme of these tracers can be set to be different
!! to that used by active tracers.


end module dyed_obc_tracer
