!> Vertical interpolation for regridding
module regrid_interp

! This file is part of MOM6. See LICENSE.md for the license.

use MOM_error_handler, only : MOM_error, FATAL
use MOM_string_functions, only : uppercase

use regrid_edge_values, only : edge_values_explicit_h2, edge_values_explicit_h4
use regrid_edge_values, only : edge_values_explicit_h4cw
use regrid_edge_values, only : edge_values_implicit_h4, edge_values_implicit_h6
use regrid_edge_values, only : edge_slopes_implicit_h3, edge_slopes_implicit_h5

use PLM_functions, only : PLM_reconstruction, PLM_boundary_extrapolation
use PPM_functions, only : PPM_reconstruction, PPM_boundary_extrapolation
use PPM_functions, only : PPM_monotonicity
use PQM_functions, only : PQM_reconstruction, PQM_boundary_extrapolation_v1

use P1M_functions, only : P1M_interpolation, P1M_boundary_extrapolation
use P3M_functions, only : P3M_interpolation, P3M_boundary_extrapolation

implicit none ; private

!> Control structure for regrid_interp module
type, public :: interp_CS_type ; private

  !> The following parameter is only relevant when used with the target
  !! interface densities regridding scheme. It indicates which interpolation
  !! to use to determine the grid.
  integer :: interpolation_scheme = -1

  !> Indicate whether high-order boundary extrapolation should be used within
  !! boundary cells
  logical :: boundary_extrapolation

  !> The vintage of the expressions to use for regridding
  integer :: answer_date = 99991231
end type interp_CS_type

public regridding_set_ppolys, build_and_interpolate_grid
public set_interp_scheme, set_interp_extrap, set_interp_answer_date

! List of interpolation schemes
integer, parameter :: INTERPOLATION_P1M_H2     = 0 !< O(h^2)
integer, parameter :: INTERPOLATION_P1M_H4     = 1 !< O(h^2)
integer, parameter :: INTERPOLATION_P1M_IH4    = 2 !< O(h^2)
integer, parameter :: INTERPOLATION_PLM        = 3 !< O(h^2)
integer, parameter :: INTERPOLATION_PPM_CW     =10 !< O(h^3)
integer, parameter :: INTERPOLATION_PPM_H4     = 4 !< O(h^3)
integer, parameter :: INTERPOLATION_PPM_IH4    = 5 !< O(h^3)
integer, parameter :: INTERPOLATION_P3M_IH4IH3 = 6 !< O(h^4)
integer, parameter :: INTERPOLATION_P3M_IH6IH5 = 7 !< O(h^4)
integer, parameter :: INTERPOLATION_PQM_IH4IH3 = 8 !< O(h^4)
integer, parameter :: INTERPOLATION_PQM_IH6IH5 = 9 !< O(h^5)

!>@{ Interpolant degrees
integer, parameter :: DEGREE_1 = 1, DEGREE_2 = 2, DEGREE_3 = 3, DEGREE_4 = 4
integer, public, parameter :: DEGREE_MAX = 5
!>@}

!> When the N-R algorithm produces an estimate that lies outside [0,1], the
!! estimate is set to be equal to the boundary location, 0 or 1, plus or minus
!! an offset, respectively, when the derivative is zero at the boundary [nondim].
real, public, parameter    :: NR_OFFSET = 1e-6
!> Maximum number of Newton-Raphson iterations. Newton-Raphson iterations are
!! used to build the new grid by finding the coordinates associated with
!! target densities and interpolations of degree larger than 1.
integer, public, parameter :: NR_ITERATIONS = 8
!> Tolerance for Newton-Raphson iterations (stop when increment falls below this) [nondim]
real, public, parameter    :: NR_TOLERANCE = 1e-12

contains

!> Builds an interpolated profile for the densities within each grid cell.
!!
!! It may happen that, given a high-order interpolator, the number of
!! available layers is insufficient (e.g., there are two available layers for
!! a third-order PPM ih4 scheme). In these cases, we resort to the simplest
!! continuous linear scheme (P1M h2).
subroutine regridding_set_ppolys(CS, densities, n0, h0, ppoly0_E, ppoly0_S, &
               ppoly0_coefs, degree, h_neglect, h_neglect_edge)
  type(interp_CS_type),  intent(in)    :: CS !< Interpolation control structure
  integer,               intent(in)    :: n0 !< Number of cells on source grid
  real, dimension(n0),   intent(in)    :: densities !< Actual cell densities [A]
  real, dimension(n0),   intent(in)    :: h0 !< cell widths on source grid [H]
  real, dimension(n0,2), intent(inout) :: ppoly0_E  !< Edge value of polynomial [A]
  real, dimension(n0,2), intent(inout) :: ppoly0_S  !< Edge slope of polynomial [A H-1]
  real, dimension(n0,DEGREE_MAX+1), intent(inout) :: ppoly0_coefs !< Coefficients of polynomial [A]
  integer,               intent(inout) :: degree    !< The degree of the polynomials
  real,                  intent(in)    :: h_neglect !< A negligibly small width for the
                                             !! purpose of cell reconstructions [H]
                                             !! in the same units as h0.
  real,        optional, intent(in)    :: h_neglect_edge !< A negligibly small width
                                             !! for the purpose of edge value calculations [H]
                                             !! in the same units as h0.
  ! Local variables
  real :: h_neg_edge  ! A negligibly small width for the purpose of edge value
                      ! calculations in the same units as h0 [H]
  logical :: extrapolate

  h_neg_edge = h_neglect ; if (present(h_neglect_edge)) h_neg_edge = h_neglect_edge

  ! Reset piecewise polynomials
  ppoly0_E(:,:) = 0.0
  ppoly0_S(:,:) = 0.0
  ppoly0_coefs(:,:) = 0.0

  extrapolate = CS%boundary_extrapolation

  ! Compute the interpolated profile of the density field and build grid
  select case (CS%interpolation_scheme)

    case ( INTERPOLATION_P1M_H2 )
      degree = DEGREE_1
      call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
      call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
      if (extrapolate) then
        call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
      endif

    case ( INTERPOLATION_P1M_H4 )
      degree = DEGREE_1
      if ( n0 >= 4 ) then
        call edge_values_explicit_h4( n0, h0, densities, ppoly0_E, h_neg_edge, answer_date=CS%answer_date )
      else
        call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
      endif
      call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
      if (extrapolate) then
        call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
      endif

    case ( INTERPOLATION_P1M_IH4 )
      degree = DEGREE_1
      if ( n0 >= 4 ) then
        call edge_values_implicit_h4( n0, h0, densities, ppoly0_E, h_neg_edge, answer_date=CS%answer_date )
      else
        call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
      endif
      call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
      if (extrapolate) then
        call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
      endif

    case ( INTERPOLATION_PLM )
      degree = DEGREE_1
      call PLM_reconstruction( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect )
      if (extrapolate) then
        call PLM_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect )
      endif

    case ( INTERPOLATION_PPM_CW )
      if ( n0 >= 4 ) then
        degree = DEGREE_2
        call edge_values_explicit_h4cw( n0, h0, densities, ppoly0_E, h_neg_edge )
        call PPM_monotonicity(   n0,     densities, ppoly0_E )
        call PPM_reconstruction( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call PPM_boundary_extrapolation( n0, h0, densities, ppoly0_E, &
                                           ppoly0_coefs, h_neglect )
        endif
      else
        degree = DEGREE_1
        call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
        call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
        endif
      endif

    case ( INTERPOLATION_PPM_H4 )
      if ( n0 >= 4 ) then
        degree = DEGREE_2
        call edge_values_explicit_h4( n0, h0, densities, ppoly0_E, h_neg_edge, answer_date=CS%answer_date )
        call PPM_reconstruction( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call PPM_boundary_extrapolation( n0, h0, densities, ppoly0_E, &
                                           ppoly0_coefs, h_neglect )
        endif
      else
        degree = DEGREE_1
        call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
        call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
        endif
      endif

    case ( INTERPOLATION_PPM_IH4 )
      if ( n0 >= 4 ) then
        degree = DEGREE_2
        call edge_values_implicit_h4( n0, h0, densities, ppoly0_E, h_neg_edge, answer_date=CS%answer_date )
        call PPM_reconstruction( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call PPM_boundary_extrapolation( n0, h0, densities, ppoly0_E, &
                                           ppoly0_coefs, h_neglect )
        endif
      else
        degree = DEGREE_1
        call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
        call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
        endif
      endif

    case ( INTERPOLATION_P3M_IH4IH3 )
      if ( n0 >= 4 ) then
        degree = DEGREE_3
        call edge_values_implicit_h4( n0, h0, densities, ppoly0_E, h_neg_edge, answer_date=CS%answer_date )
        call edge_slopes_implicit_h3( n0, h0, densities, ppoly0_S, h_neglect, answer_date=CS%answer_date )
        call P3M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_S, &
                                ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call P3M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_S, &
                                           ppoly0_coefs, h_neglect, h_neg_edge )
        endif
      else
        degree = DEGREE_1
        call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
        call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
        endif
      endif

    case ( INTERPOLATION_P3M_IH6IH5 )
      if ( n0 >= 6 ) then
        degree = DEGREE_3
        call edge_values_implicit_h6( n0, h0, densities, ppoly0_E, h_neg_edge, answer_date=CS%answer_date )
        call edge_slopes_implicit_h5( n0, h0, densities, ppoly0_S, h_neglect, answer_date=CS%answer_date )
        call P3M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_S, &
                                ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call P3M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_S, &
                   ppoly0_coefs, h_neglect, h_neglect_edge )
        endif
      else
        degree = DEGREE_1
        call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
        call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
        endif
      endif

    case ( INTERPOLATION_PQM_IH4IH3 )
      if ( n0 >= 4 ) then
        degree = DEGREE_4
        call edge_values_implicit_h4( n0, h0, densities, ppoly0_E, h_neg_edge, answer_date=CS%answer_date )
        call edge_slopes_implicit_h3( n0, h0, densities, ppoly0_S, h_neglect, answer_date=CS%answer_date )
        call PQM_reconstruction( n0, h0, densities, ppoly0_E, ppoly0_S, &
                                 ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call PQM_boundary_extrapolation_v1( n0, h0, densities, ppoly0_E, ppoly0_S, &
                                 ppoly0_coefs, h_neglect )
        endif
      else
        degree = DEGREE_1
        call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
        call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
        endif
      endif

    case ( INTERPOLATION_PQM_IH6IH5 )
      if ( n0 >= 6 ) then
        degree = DEGREE_4
        call edge_values_implicit_h6( n0, h0, densities, ppoly0_E, h_neg_edge, answer_date=CS%answer_date )
        call edge_slopes_implicit_h5( n0, h0, densities, ppoly0_S, h_neglect, answer_date=CS%answer_date )
        call PQM_reconstruction( n0, h0, densities, ppoly0_E, ppoly0_S, &
                                 ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call PQM_boundary_extrapolation_v1( n0, h0, densities, ppoly0_E, ppoly0_S, &
                                 ppoly0_coefs, h_neglect )
        endif
      else
        degree = DEGREE_1
        call edge_values_explicit_h2( n0, h0, densities, ppoly0_E )
        call P1M_interpolation( n0, h0, densities, ppoly0_E, ppoly0_coefs, h_neglect, answer_date=CS%answer_date )
        if (extrapolate) then
          call P1M_boundary_extrapolation( n0, h0, densities, ppoly0_E, ppoly0_coefs )
        endif
      endif
  end select

end subroutine regridding_set_ppolys

!> Given target values (e.g., density), build new grid based on polynomial
!!
!! Given the grid 'grid0' and the piecewise polynomial interpolant
!! 'ppoly0' (possibly discontinuous), the coordinates of the new grid 'grid1'
!! are determined by finding the corresponding target interface densities.
subroutine interpolate_grid( n0, h0, x0, ppoly0_E, ppoly0_coefs, &
                             target_values, degree, n1, h1, x1, answer_date )
  integer,               intent(in)     :: n0            !< Number of points on source grid
  integer,               intent(in)     :: n1            !< Number of points on target grid
  real, dimension(n0),   intent(in)     :: h0            !< Thicknesses of source grid cells [H]
  real, dimension(n0+1), intent(in)     :: x0            !< Source interface positions [H]
  real, dimension(n0,2), intent(in)     :: ppoly0_E      !< Edge values of interpolating polynomials [A]
  real, dimension(n0,DEGREE_MAX+1), &
                          intent(in)    :: ppoly0_coefs  !< Coefficients of interpolating polynomials [A]
  real, dimension(n1+1),  intent(in)    :: target_values !< Target values of interfaces [A]
  integer,                intent(in)    :: degree        !< Degree of interpolating polynomials
  real, dimension(n1),    intent(inout) :: h1            !< Thicknesses of target grid cells [H]
  real, dimension(n1+1),  intent(inout) :: x1            !< Target interface positions [H]
  integer,      optional, intent(in)    :: answer_date   !< The vintage of the expressions to use

  ! Local variables
  integer        :: k ! loop index
  real           :: t ! current interface target density [A]

  ! Make sure boundary coordinates of new grid coincide with boundary
  ! coordinates of previous grid
  x1(1) = x0(1)
  x1(n1+1) = x0(n0+1)

  ! Find coordinates for interior target values
  do k = 2,n1
    t = target_values(k)
    x1(k) = get_polynomial_coordinate ( n0, h0, x0, ppoly0_E, ppoly0_coefs, t, degree, &
                                        answer_date=answer_date )
    h1(k-1) = x1(k) - x1(k-1)
  enddo
  h1(n1) = x1(n1+1) - x1(n1)

end subroutine interpolate_grid

!> Build a grid by interpolating for target values
subroutine build_and_interpolate_grid(CS, densities, n0, h0, x0, target_values, &
                                      n1, h1, x1, h_neglect, h_neglect_edge)
  type(interp_CS_type),  intent(in)    :: CS  !< A control structure for regrid_interp
  integer,               intent(in)    :: n0  !< The number of points on the input grid
  integer,               intent(in)    :: n1  !< The number of points on the output grid
  real, dimension(n0),   intent(in)    :: densities !< Input cell densities [R ~> kg m-3]
  real, dimension(n1+1), intent(in)    :: target_values !< Target values of interfaces [R ~> kg m-3]
  real, dimension(n0),   intent(in)    :: h0  !< Initial cell widths [H]
  real, dimension(n0+1), intent(in)    :: x0  !< Source interface positions [H]
  real, dimension(n1),   intent(inout) :: h1  !< Output cell widths [H]
  real, dimension(n1+1), intent(inout) :: x1  !< Target interface positions [H]
  real,                  intent(in)    :: h_neglect !< A negligibly small width for the
                                           !! purpose of cell reconstructions [H]
                                           !! in the same units as h0.
  real,        optional, intent(in)    :: h_neglect_edge !< A negligibly small width
                                           !! for the purpose of edge value calculations [H]
                                           !! in the same units as h0.

  real, dimension(n0,2) :: ppoly0_E   ! Polynomial edge values [R ~> kg m-3]
  real, dimension(n0,2) :: ppoly0_S   ! Polynomial edge slopes [R H-1]
  real, dimension(n0,DEGREE_MAX+1) :: ppoly0_C  ! Polynomial interpolant coeficients on the local 0-1 grid [R ~> kg m-3]
  integer :: degree

  call regridding_set_ppolys(CS, densities, n0, h0, ppoly0_E, ppoly0_S, ppoly0_C, &
       degree, h_neglect, h_neglect_edge)
  call interpolate_grid(n0, h0, x0, ppoly0_E, ppoly0_C, target_values, degree, &
       n1, h1, x1, answer_date=CS%answer_date)
end subroutine build_and_interpolate_grid

!> Given a target value, find corresponding coordinate for given polynomial
!!
!! Here, 'ppoly' is assumed to be a piecewise discontinuous polynomial of degree
!! 'degree' throughout the domain defined by 'grid'. A target value is given
!! and we need to determine the corresponding grid coordinate to define the
!! new grid.
!!
!! If the target value is out of range, the grid coordinate is simply set to
!! be equal to one of the boundary coordinates, which results in vanished layers
!! near the boundaries.
!!
!! IT IS ASSUMED THAT THE PIECEWISE POLYNOMIAL IS MONOTONICALLY INCREASING.
!! IF THIS IS NOT THE CASE, THE NEW GRID MAY BE ILL-DEFINED.
!!
!! It is assumed that the number of cells defining 'grid' and 'ppoly' are the
!! same.
function get_polynomial_coordinate( N, h, x_g, edge_values, ppoly_coefs, &
                                    target_value, degree, answer_date ) result ( x_tgt )
  ! Arguments
  integer,              intent(in) :: N            !< Number of grid cells
  real, dimension(N),   intent(in) :: h            !< Grid cell thicknesses [H]
  real, dimension(N+1), intent(in) :: x_g          !< Grid interface locations [H]
  real, dimension(N,2), intent(in) :: edge_values  !< Edge values of interpolating polynomials [A]
  real, dimension(N,DEGREE_MAX+1), intent(in) :: ppoly_coefs  !< Coefficients of interpolating polynomials [A]
  real,                 intent(in) :: target_value !< Target value to find position for [A]
  integer,              intent(in) :: degree       !< Degree of the interpolating polynomials
  integer,              intent(in) :: answer_date  !< The vintage of the expressions to use
  real                             :: x_tgt        !< The position of x_g at which target_value is found [H]

  ! Local variables
  real                        :: xi0         ! normalized target coordinate [nondim]
  real, dimension(DEGREE_MAX) :: a           ! polynomial coefficients [A]
  real                        :: numerator   ! The numerator of an expression [A]
  real                        :: denominator ! The denominator of an expression [A]
  real                        :: delta       ! Newton-Raphson increment [nondim]
!   real                        :: x           ! global target coordinate [nondim]
  real                        :: eps         ! offset used to get away from boundaries [nondim]
  real                        :: grad        ! gradient during N-R iterations [A]
  integer :: i, k, iter  ! loop indices
  integer :: k_found     ! index of target cell
  character(len=320) :: mesg
  logical :: use_2018_answers  ! If true use older, less accurate expressions.

  eps = NR_OFFSET
  k_found = -1
  use_2018_answers = (answer_date < 20190101)

  ! If the target value is outside the range of all values, we
  ! force the target coordinate to be equal to the lowest or
  ! largest value, depending on which bound is overtaken
  if ( target_value <= edge_values(1,1) ) then
    x_tgt = x_g(1)
    return  ! return because there is no need to look further
  endif

  ! Since discontinuous edge values are allowed, we check whether the target
  ! value lies between two discontinuous edge values at interior interfaces
  do k = 2,N
    if ( ( target_value >= edge_values(k-1,2) ) .AND. ( target_value <= edge_values(k,1) ) ) then
      x_tgt = x_g(k)
      return   ! return because there is no need to look further
    endif
  enddo

  ! If the target value is outside the range of all values, we
  ! force the target coordinate to be equal to the lowest or
  ! largest value, depending on which bound is overtaken
  if ( target_value >= edge_values(N,2) ) then
    x_tgt = x_g(N+1)
    return  ! return because there is no need to look further
  endif

  ! At this point, we know that the target value is bounded and does not
  ! lie between discontinuous, monotonic edge values. Therefore,
  ! there is a unique solution. We loop on all cells and find which one
  ! contains the target value. The variable k_found holds the index value
  ! of the cell where the taregt value lies.
  do k = 1,N
    if ( ( target_value > edge_values(k,1) ) .AND. ( target_value < edge_values(k,2) ) ) then
      k_found = k
      exit
    endif
  enddo

  ! At this point, 'k_found' should be strictly positive. If not, this is
  ! a major failure because it means we could not find any target cell
  ! despite the fact that the target value lies between the extremes. It
  ! means there is a major problem with the interpolant. This needs to be
  ! reported.
  if ( k_found == -1 ) then
    write(mesg,*) 'Could not find target coordinate', target_value, 'in get_polynomial_coordinate. This is '//&
                  'caused by an inconsistent interpolant (perhaps not monotonically increasing):', &
                  target_value, edge_values(1,1), edge_values(N,2)
    call MOM_error( FATAL, mesg )
  endif

  ! Reset all polynomial coefficients to 0 and copy those pertaining to
  ! the found cell
  a(:) = 0.0
  do i = 1,degree+1
    a(i) = ppoly_coefs(k_found,i)
  enddo

  ! Guess the middle of the cell to start Newton-Raphson iterations
  xi0 = 0.5

  ! Newton-Raphson iterations
  do iter = 1,NR_ITERATIONS

    if (use_2018_answers) then
      numerator = a(1) + a(2)*xi0 + a(3)*xi0*xi0 + a(4)*xi0*xi0*xi0 + &
                  a(5)*xi0*xi0*xi0*xi0 - target_value
      denominator = a(2) + 2*a(3)*xi0 + 3*a(4)*xi0*xi0 + 4*a(5)*xi0*xi0*xi0
    else  ! These expressions are mathematicaly equivalent but more accurate.
      numerator = (a(1) - target_value) + xi0*(a(2) + xi0*(a(3) + xi0*(a(4) + a(5)*xi0)))
      denominator = a(2) + xi0*(2.*a(3) + xi0*(3.*a(4) + 4.*a(5)*xi0))
    endif

    delta = -numerator / denominator

    xi0 = xi0 + delta

    ! Check whether new estimate is out of bounds. If the new estimate is
    ! indeed out of bounds, we manually set it to be equal to the overtaken
    ! bound with a small offset towards the interior when the gradient of
    ! the function at the boundary is zero (in which case, the Newton-Raphson
    ! algorithm does not converge).
    if ( xi0 < 0.0 ) then
      xi0 = 0.0
      grad = a(2)
      if ( grad == 0.0 ) xi0 = xi0 + eps
    endif

    if ( xi0 > 1.0 ) then
      xi0 = 1.0
      if (use_2018_answers) then
        grad = a(2) + 2*a(3) + 3*a(4) + 4*a(5)
      else  ! These expressions are mathematicaly equivalent but more accurate.
        grad = a(2) + (2.*a(3) + (3.*a(4) + 4.*a(5)))
      endif
      if ( grad == 0.0 ) xi0 = xi0 - eps
    endif

    ! break if converged or too many iterations taken
    if ( abs(delta) < NR_TOLERANCE ) exit
  enddo ! end Newton-Raphson iterations

  x_tgt = x_g(k_found) + xi0 * h(k_found)
end function get_polynomial_coordinate

!> Numeric value of interpolation_scheme corresponding to scheme name
integer function interpolation_scheme(interp_scheme)
  character(len=*), intent(in) :: interp_scheme !< Name of the interpolation scheme
        !! Valid values include "P1M_H2", "P1M_H4", "P1M_IH2", "PLM", "PPM_CW", "PPM_H4",
        !!   "PPM_IH4", "P3M_IH4IH3", "P3M_IH6IH5", "PQM_IH4IH3", and "PQM_IH6IH5"

  select case ( uppercase(trim(interp_scheme)) )
    case ("P1M_H2");     interpolation_scheme = INTERPOLATION_P1M_H2
    case ("P1M_H4");     interpolation_scheme = INTERPOLATION_P1M_H4
    case ("P1M_IH2");    interpolation_scheme = INTERPOLATION_P1M_IH4
    case ("PLM");        interpolation_scheme = INTERPOLATION_PLM
    case ("PPM_CW");     interpolation_scheme = INTERPOLATION_PPM_CW
    case ("PPM_H4");     interpolation_scheme = INTERPOLATION_PPM_H4
    case ("PPM_IH4");    interpolation_scheme = INTERPOLATION_PPM_IH4
    case ("P3M_IH4IH3"); interpolation_scheme = INTERPOLATION_P3M_IH4IH3
    case ("P3M_IH6IH5"); interpolation_scheme = INTERPOLATION_P3M_IH6IH5
    case ("PQM_IH4IH3"); interpolation_scheme = INTERPOLATION_PQM_IH4IH3
    case ("PQM_IH6IH5"); interpolation_scheme = INTERPOLATION_PQM_IH6IH5
    case default ; call MOM_error(FATAL, "regrid_interp: "//&
     "Unrecognized choice for INTERPOLATION_SCHEME ("//trim(interp_scheme)//").")
  end select
end function interpolation_scheme

!> Store the interpolation_scheme value in the interp_CS based on the input string.
subroutine set_interp_scheme(CS, interp_scheme)
  type(interp_CS_type), intent(inout) :: CS  !< A control structure for regrid_interp
  character(len=*),     intent(in) :: interp_scheme !< Name of the interpolation scheme
        !! Valid values include "P1M_H2", "P1M_H4", "P1M_IH2", "PLM", "PPM_CW", "PPM_H4",
        !!   "PPM_IH4", "P3M_IH4IH3", "P3M_IH6IH5", "PQM_IH4IH3", and "PQM_IH6IH5"

  CS%interpolation_scheme = interpolation_scheme(interp_scheme)
end subroutine set_interp_scheme

!> Store the boundary_extrapolation value in the interp_CS
subroutine set_interp_extrap(CS, extrap)
  type(interp_CS_type), intent(inout) :: CS  !< A control structure for regrid_interp
  logical,              intent(in)    :: extrap !< Indicate whether high-order boundary
                                             !! extrapolation should be used in boundary cells

  CS%boundary_extrapolation = extrap
end subroutine set_interp_extrap

!> Store the value of the answer_date in the interp_CS
subroutine set_interp_answer_date(CS, answer_date)
  type(interp_CS_type), intent(inout) :: CS  !< A control structure for regrid_interp
  integer,              intent(in)    :: answer_date !< An integer encoding the vintage of
                                             !! the expressions to use for regridding

  CS%answer_date = answer_date
end subroutine set_interp_answer_date

end module regrid_interp
