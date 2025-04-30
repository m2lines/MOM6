program time_MOM_ANN

! This file is part of MOM6. See LICENSE.md for the license.

use MOM_ANN, only : ANN_CS
use MOM_ANN, only : ANN_allocate, ANN_apply, ANN_end
use MOM_ANN, only : set_layer

implicit none

real :: timing
real :: tmin, tmax, tmean, tstd
integer, parameter :: nits = 1000000
integer, parameter :: nsamp = 20
integer, parameter :: nxy = 100
integer :: samp

tmin = 1e9
tmax = 0.
tmean = 0.
tstd = 0.
do samp = 1, nsamp
  call run_scalar(7, 4, 16, 1, nits, timing)
  tmin = min( tmin, timing )
  tmax = max( tmax, timing )
  tmean = tmean + timing
  tstd = tstd + timing**2
enddo
tmean = tmean / real(nsamp)
tstd = tstd / real(nsamp) ! convert to mean of squares
tstd = tstd - tmean**2  ! convert to variance
tstd = sqrt( tstd * real(nsamp) / real(nsamp-1) ) ! convert to standard deviation

! Display results in YAML
write(*,'(a)') "{"
write(*,"(2x,5a)") '"MOM_ANN ANN_apply(scalar)": {'
write(*,"(4x,a,1pe11.4,',')") '"min": ',tmin
write(*,"(4x,a,1pe11.4,',')") '"mean":',tmean
write(*,"(4x,a,1pe11.4,',')") '"std": ',tstd
write(*,"(4x,a,i0,',')") '"n_samples": ',nsamp*nits
write(*,"(4x,a,1pe11.4,'},')") '"max": ',tmax

tmin = 1e9
tmax = 0.
tmean = 0.
tstd = 0.
do samp = 1, nsamp
  call run_array(7, 4, 16, 1, nits/nxy, nxy, timing)
  tmin = min( tmin, timing )
  tmax = max( tmax, timing )
  tmean = tmean + timing
  tstd = tstd + timing**2
enddo
tmean = tmean / real(nsamp)
tstd = tstd / real(nsamp) ! convert to mean of squares
tstd = tstd - tmean**2  ! convert to variance
tstd = sqrt( tstd * real(nsamp) / real(nsamp-1) ) ! convert to standard deviation

! Display results in YAML
write(*,"(2x,5a)") '"MOM_ANN ANN_apply(scalar)": {'
write(*,"(4x,a,1pe11.4,',')") '"min": ',tmin
write(*,"(4x,a,1pe11.4,',')") '"mean":',tmean
write(*,"(4x,a,1pe11.4,',')") '"std": ',tstd
write(*,"(4x,a,i0,',')") '"n_samples": ',nsamp*nits
write(*,"(4x,a,1pe11.4,'}')") '"max": ',tmax
write(*,'(a)') "}"

contains

!> Time ANN inference on scalar inputs
subroutine run_scalar(nlayers, nin, width, nout, nits, timing)
  integer, intent(in)  :: nlayers      !< Number of layers
  integer, intent(in)  :: nin          !< Number of inputs
  integer, intent(in)  :: width        !< Width of hidden layers
  integer, intent(in)  :: nout         !< Number of outputs
  integer, intent(in)  :: nits         !< Number of calls to time
  real,    intent(out) :: timing       !< The average time taken for nits calls [seconds]
  ! Local variables
  type(ANN_CS) :: ANN ! ANN
  integer :: widths(nlayers) ! Width of each layer
  real :: x(nin) ! Inputs
  real :: y(nin) ! Outputs
  real :: start, finish ! CPU times [s]
  integer :: iter ! Loop counter

  widths(:) = width
  widths(1) = nin
  widths(nlayers) = nout

  call random_ANN(ANN, nlayers, widths)
  call random_number(x)

  call cpu_time(start)
  do iter = 1, nits ! Make many passes to reduce sampling error
    call ANN_apply(x, y, ANN)
  enddo
  call cpu_time(finish)

  timing = (finish-start)/real(nits) ! Average time per call

end subroutine run_scalar

!> Time ANN inference on 2d array inputs
subroutine run_array(nlayers, nin, width, nout, nits, nxy, timing)
  integer, intent(in)  :: nlayers      !< Number of layers
  integer, intent(in)  :: nin          !< Number of inputs
  integer, intent(in)  :: width        !< Width of hidden layers
  integer, intent(in)  :: nout         !< Number of outputs
  integer, intent(in)  :: nits         !< Number of calls to time
  integer, intent(in)  :: nxy          !< Spatial dimension
  real,    intent(out) :: timing       !< The average time taken for nits calls [seconds]
  ! Local variables
  type(ANN_CS) :: ANN ! ANN
  integer :: widths(nlayers) ! Width of each layer
  real :: x(nin,nxy) ! Inputs
  real :: y(nin,nxy) ! Outputs
  real :: start, finish ! CPU times [s]
  integer :: iter ! Loop counter
  integer :: ij ! Horizontal loop index

  widths(:) = width
  widths(1) = nin
  widths(nlayers) = nout

  call random_ANN(ANN, nlayers, widths)
  call random_number(x)

  call cpu_time(start)
  do iter = 1, nits ! Make many passes to reduce sampling error
    do ij = 1, nxy
      call ANN_apply(x(:,ij), y(:,ij), ANN)
    enddo
  enddo
  call cpu_time(finish)

  timing = (finish-start)/real(nits) ! Average time per call

end subroutine run_array

!> Create a random ANN
subroutine random_ANN(ANN, nlayers, widths)
  type(ANN_CS), intent(inout) :: ANN !< ANN control structure
  integer,      intent(in)    :: nlayers !< Number of layers
  integer,      intent(in)    :: widths(nlayers) !< Width of each layer
  ! Local variables
  integer :: l

  call ANN_allocate(ANN, nlayers, widths)

  do l = 1, nlayers-1
    call randomize_layer(ANN, nlayers, l, widths)
  enddo

end subroutine random_ANN

!> Fill a layer with random numbers
subroutine randomize_layer(ANN, nlayers, layer, widths)
  type(ANN_CS), intent(inout) :: ANN !< ANN control structure
  integer,      intent(in)    :: nlayers !< Number of layers
  integer,      intent(in)    :: layer !< Layer number to randomize
  integer,      intent(in)    :: widths(nlayers) !< Width of each layer
  ! Local variables
  real :: weights(widths(layer+1),widths(layer)) ! Weights
  real :: biases(widths(layer+1)) ! Biases

  call random_number(weights)
  weights(:,:) = 2. * weights(:,:) - 1.

  call random_number(biases)
  biases(:) = 2. * biases(:) - 1.

  call set_layer(ANN, layer, weights, biases, layer<nlayers-1)

end subroutine randomize_layer

end program time_MOM_ANN
