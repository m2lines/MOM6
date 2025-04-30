!> Implements the general purpose ANN.
module MOM_ANN

! This file is part of MOM6. See LICENSE.md for the license

use MOM_io, only : MOM_read_data, field_exists
use MOM_error_handler, only : MOM_error, FATAL, MOM_mesg
use numerical_testing_type, only : testing

implicit none ; private

!#include <MOM_memory.h>

public ANN_init, ANN_allocate, ANN_apply, ANN_end, ANN_unit_tests
public set_layer, set_input_normalization, set_output_normalization

!> Type for a single Linear layer of ANN,
!! i.e. stores the matrix A and bias b
!! for matrix-vector multiplication
!! y = A*x + b.
type, private :: layer_type; private
  integer :: output_width        !< Number of rows in matrix A
  integer :: input_width         !< Number of columns in matrix A
  logical :: activation = .True. !< If true, apply the default activation function

  real, allocatable :: A(:,:) !< Matrix in column-major order
                              !! of size A(output_width, input_width) [nondim]
  real, allocatable :: b(:)   !< bias vector of size output_width [nondim]
end type layer_type

!> Control structure/type for ANN
type, public :: ANN_CS ; private
  ! Parameters
  integer :: num_layers          !< Number of layers in the ANN, including the input and output.
                                 !! For example, for ANN with one hidden layer, num_layers = 3.
  integer, allocatable &
          :: layer_sizes(:)      !< Array of length num_layers, storing the number of neurons in
                                 !! each layer.

  type(layer_type), allocatable &
          :: layers(:)           !< Array of length num_layers-1, where each element is the Linear
                                 !! transformation between layers defined by Matrix A and vias b.

  real, allocatable :: &
    input_means(:), &  !< Array of length layer_sizes(1) containing the mean of each input feature
                       !! prior to normalization by input_norms [arbitrary].
    input_norms(:), &  !< Array of length layer_sizes(1) containing the *inverse* of the standard
                       !! deviation for each input feature used to normalize (multiply) before
                       !! feeding into the ANN [arbitrary]
    output_means(:), & !< Array of length layer_sizes(num_layers) containing the mean of each
                       !! output prior to normalization by output_norms [arbitrary].
    output_norms(:)    !< Array of length layer_sizes(num_layers) containing the standard deviation
                       !! each output of the ANN will be multiplied [arbitrary]
end type ANN_CS

contains

!> Initialization of ANN. Allocates memory and reads ANN parameters from NetCDF file.
!! The NetCDF file must contain:
!! Integer num_layers.
!! Integer arrays: layer_sizes, input_norms, output_norms
!! Matrices and biases for Linear layers can be Real(4) or Real(8) and
!! are named as: A0, b0 for the first layer; A1, b1 for the second layer and so on.
subroutine ANN_init(CS, NNfile)
  type(ANN_CS), intent(inout)  :: CS     !< ANN control structure.
  character(*), intent(in)     :: NNfile !< The name of NetCDF file having neural network parameters
  ! Local variables
  integer :: i
  integer :: num_layers ! Number of layers, including input and output layers
  integer, allocatable :: layer_sizes(:) ! Number of neurons in each layer
  character(len=1) :: layer_num_str
  character(len=3) :: fieldname

  call MOM_mesg('ANN: init from ' // trim(NNfile), 2)

  ! Read the number of layers
  call MOM_read_data(NNfile, "num_layers", num_layers)

  ! Read size of layers
  allocate( layer_sizes(num_layers) )
  call MOM_read_data(NNfile, "layer_sizes", layer_sizes)

  ! Allocates the memory for storing normalization, weights and biases
  call ANN_allocate(CS, num_layers, layer_sizes)
  deallocate( layer_sizes )

  ! Read normalization factors
  if (field_exists(NNfile, 'input_means')) &
    call MOM_read_data(NNfile, 'input_means', CS%input_means)
  if (field_exists(NNfile, 'input_norms')) then
    call MOM_read_data(NNfile, 'input_norms', CS%input_norms)
    ! We calculate the reciprocal here to avoid repeated divisions later
    CS%input_norms(:) = 1.  / CS%input_norms(:)
  endif
  if (field_exists(NNfile, 'output_means')) &
    call MOM_read_data(NNfile, 'output_means', CS%output_means)
  if (field_exists(NNfile, 'output_norms')) &
    call MOM_read_data(NNfile, 'output_norms', CS%output_norms)

  ! Allocate and read matrix A and bias b for each layer
  do i = 1,CS%num_layers-1
    CS%layers(i)%input_width = CS%layer_sizes(i)
    CS%layers(i)%output_width = CS%layer_sizes(i+1)

    ! Reading matrix A
    write(layer_num_str, '(I0)') i-1
    fieldname = trim('A') // trim(layer_num_str)
    call MOM_read_data(NNfile, fieldname, CS%layers(i)%A, &
                        (/1,1,1,1/),(/CS%layers(i)%output_width,CS%layers(i)%input_width,1,1/))

    ! Reading bias b
    fieldname = trim('b') // trim(layer_num_str)
    call MOM_read_data(NNfile, fieldname, CS%layers(i)%b)
  enddo

  ! No activation function for the last layer
  CS%layers(CS%num_layers-1)%activation = .False.

  if (field_exists(NNfile, 'x_test') .and. field_exists(NNfile, 'y_test') ) &
  call ANN_test(CS, NNfile)

  call MOM_mesg('ANN: have been read from ' // trim(NNfile), 2)

end subroutine ANN_init

!> Allocate an ANN
!!
!! This creates the memory for storing weights and intermediate work arrays, but does not set
!! the values of weights or biases (not even initializing with zeros).
subroutine ANN_allocate(CS, num_layers, layer_sizes)
  type(ANN_CS), intent(inout) :: CS !< ANN control structure
  integer,      intent(in)    :: num_layers !< The number of layers, including the input and output layer
  integer,      intent(in)    :: layer_sizes(num_layers) !< The number of neurons in each layer
  ! Local variables
  integer :: l ! Layer number

  ! Assert that there is always an input and output layer
  if (num_layers < 2) call MOM_error(FATAL, "The number of layers in an ANN must be >=2")

  CS%num_layers = num_layers

  ! Layers
  allocate( CS%layer_sizes(CS%num_layers) )
  CS%layer_sizes(:) = layer_sizes(:)

  ! Input and output normalization values
  allocate( CS%input_means(CS%layer_sizes(1)), source=0. ) ! Assume zero mean by default
  allocate( CS%input_norms(CS%layer_sizes(1)), source=1. ) ! Assume unit variance by default
  allocate( CS%output_means(CS%layer_sizes(CS%num_layers)), source=0. ) ! Assume zero mean by default
  allocate( CS%output_norms(CS%layer_sizes(CS%num_layers)), source=1. ) ! Assume unit variance by default

  ! Allocate the Linear transformations between layers
  allocate(CS%layers(CS%num_layers-1))

  ! Allocate matrix A and bias b for each layer
  do l = 1, CS%num_layers-1
    CS%layers(l)%input_width = CS%layer_sizes(l)
    CS%layers(l)%output_width = CS%layer_sizes(l+1)

    allocate( CS%layers(l)%A(CS%layers(l)%output_width, CS%layers(l)%input_width) )
    allocate( CS%layers(l)%b(CS%layers(l)%output_width) )
  enddo

end subroutine ANN_allocate

!> Test ANN by comparing the prediction with the test data.
subroutine ANN_test(CS, NNfile)
  type(ANN_CS), intent(inout) :: CS     !< ANN control structure.
  character(*), intent(in)    :: NNfile !< The name of NetCDF file having neural network parameters
  ! Local variables
  real, dimension(:), allocatable :: x_test, y_test, y_pred ! [arbitrary]
  real :: relative_error ! [arbitrary]
  character(len=200) :: relative_error_str

  ! Allocate data
  allocate(x_test(CS%layer_sizes(1)))
  allocate(y_test(CS%layer_sizes(CS%num_layers)))
  allocate(y_pred(CS%layer_sizes(CS%num_layers)))

  ! Read test vectors
  call MOM_read_data(NNfile, 'x_test', x_test)
  call MOM_read_data(NNfile, 'y_test', y_test)

  ! Compute prediction
  call ANN_apply(x_test, y_pred, CS)

  relative_error = maxval(abs(y_pred(:) - y_test(:))) / maxval(abs(y_test(:)))

  if (relative_error > 1e-5) then
    write(relative_error_str, '(ES12.4)') relative_error
    call MOM_error(FATAL, 'Relative error in ANN prediction is too large: ' // trim(relative_error_str))
  endif

  deallocate(x_test)
  deallocate(y_test)
  deallocate(y_pred)
end subroutine ANN_test

!> Deallocates memory of ANN
subroutine ANN_end(CS)
  type(ANN_CS), intent(inout) :: CS !< ANN control structure.
  ! Local variables
  integer :: i

  deallocate(CS%layer_sizes)
  deallocate(CS%input_means)
  deallocate(CS%input_norms)
  deallocate(CS%output_means)
  deallocate(CS%output_norms)

  do i = 1, CS%num_layers-1
    deallocate(CS%layers(i)%A)
    deallocate(CS%layers(i)%b)
  enddo
  deallocate(CS%layers)

end subroutine ANN_end

!> Main ANN function: normalizes input vector x, applies Linear layers, and
!! un-normalizes the output.
subroutine ANN_apply(x, y, CS)
  type(ANN_CS), intent(in)      :: CS !< ANN control structure
  real, dimension(CS%layer_sizes(1)), &
                  intent(in)    :: x !< input [arbitrary]
  real, dimension(CS%layer_sizes(CS%num_layers)), &
                  intent(inout) :: y !< output [arbitrary]
  ! Local variables
  real, allocatable :: x_1(:), x_2(:) ! intermediate states [nondim]
  integer :: i

  ! Normalize input
  allocate(x_1(CS%layer_sizes(1)))
  do i = 1,CS%layer_sizes(1)
    x_1(i) = ( x(i) - CS%input_means(i) ) * CS%input_norms(i)
  enddo

  ! Apply Linear layers
  do i = 1, CS%num_layers-1
    allocate(x_2(CS%layer_sizes(i+1)))
    call layer_apply(x_1, x_2, CS%layers(i))
    deallocate(x_1)
    allocate(x_1(CS%layer_sizes(i+1)))
    x_1 = x_2
    deallocate(x_2)
  enddo

  ! Un-normalize output
  do i = 1, CS%layer_sizes(CS%num_layers)
    y(i) = ( x_1(i) * CS%output_norms(i) ) + CS%output_means(i)
  enddo

  deallocate(x_1)
end subroutine ANN_apply

!> The default activation function
pure function activation_fn(x) result (y)
  real, intent(in) :: x !< Scalar input value [nondim]
  real             :: y !< Scalar output value [nondim]

  y = max(x, 0.0) ! ReLU activation

end function activation_fn

!> Applies linear layer to input data x and stores the result in y with
!! y = A*x + b with optional application of the activation function.
subroutine layer_apply(x, y, layer)
  type(layer_type), intent(in)    :: layer !< Linear layer
  real, dimension(layer%input_width), &
                    intent(in)    :: x     !< Input vector [nondim]
  real, dimension(layer%output_width), &
                    intent(inout) :: y     !< Output vector [nondim]
  ! Local variables
  integer :: i, j

  y(:) = 0.
  do i=1,layer%input_width
    do j=1,layer%output_width
      ! Multiply by kernel
      y(j) = y(j) + ( x(i) * layer%A(j, i) )
    enddo
  enddo

  do j=1,layer%output_width
    ! Add bias
    y(j) = y(j) + layer%b(j)
    ! Apply activation function
    if (layer%activation) then
      y(j) = activation_fn(y(j))
    endif
  enddo
end subroutine layer_apply

!> Sets weights and bias for a single layer
subroutine set_layer(ANN, layer, weights, biases, activation)
  type(ANN_CS), intent(inout) :: ANN !< ANN control structure
  integer,      intent(in)    :: layer !< The number of the layer being adjusted
  real,         intent(in)    :: weights(:,:) !< The weights to assign
  real,         intent(in)    :: biases(:) !< The biases to assign
  logical,      intent(in)    :: activation !< Turn on the activation function

  if ( layer >= ANN%num_layers ) &
      call MOM_error(FATAL, "MOM_ANN, set_layer: layer is out of range")
  if ( layer < 1 ) &
      call MOM_error(FATAL, "MOM_ANN, set_layer: layer should be >= 1")

  if ( size(biases) /= size(ANN%layers(layer)%b) ) &
      call MOM_error(FATAL, "MOM_ANN, set_layer: mismatch in size of biases")
  ANN%layers(layer)%b(:) = biases(:)

  if ( size(weights,1) /= size(ANN%layers(layer)%A,1) ) &
      call MOM_error(FATAL, "MOM_ANN, set_layer: mismatch in size of weights (first dim)")
  if ( size(weights,2) /= size(ANN%layers(layer)%A,2) ) &
      call MOM_error(FATAL, "MOM_ANN, set_layer: mismatch in size of weights (second dim)")
  ANN%layers(layer)%A(:,:) = weights(:,:)

  ANN%layers(layer)%activation = activation
end subroutine set_layer

!> Sets input normalization
subroutine set_input_normalization(ANN, means, norms)
  type(ANN_CS),   intent(inout) :: ANN !< ANN control structure
  real, optional, intent(in)    :: means(:) !< The mean of each input
  real, optional, intent(in)    :: norms(:) !< The standard deviation of each input

  if (present(means)) then
    if ( size(means) /= size(ANN%input_means) ) &
        call MOM_error(FATAL, "MOM_ANN, set_input_normalization: mismatch in size of means")
    ANN%input_means(:) = means(:)
  endif

  if (present(norms)) then
    if ( size(norms) /= size(ANN%input_norms) ) &
        call MOM_error(FATAL, "MOM_ANN, set_input_normalization: mismatch in size of norms")
    ANN%input_norms(:) = norms(:)
  endif

end subroutine set_input_normalization

!> Sets output normalization
subroutine set_output_normalization(ANN, means, norms)
  type(ANN_CS),   intent(inout) :: ANN !< ANN control structure
  real, optional, intent(in)    :: means(:) !< The mean of each output
  real, optional, intent(in)    :: norms(:) !< The standard deviation of each output

  if (present(means)) then
    if ( size(means) /= size(ANN%output_means) ) &
        call MOM_error(FATAL, "MOM_ANN, set_output_normalization: mismatch in size of means")
    ANN%output_means(:) = means(:)
  endif

  if (present(norms)) then
    if ( size(norms) /= size(ANN%output_norms) ) &
        call MOM_error(FATAL, "MOM_ANN, set_output_normalization: mismatch in size of norms")
    ANN%output_norms(:) = norms(:)
  endif

end subroutine set_output_normalization

!> Runs unit tests on ANN functions.
!!
!! Should only be called from a single/root thread.
!! Returns True if a test fails, otherwise False.
logical function ANN_unit_tests(verbose)
  logical, intent(in) :: verbose !< If true, write results to stdout
  ! Local variables
  type(ANN_CS) :: ANN ! An ANN
  type(testing) :: test ! Manage tests
  real, allocatable :: y(:) ! Outputs [arbitrary]

  ANN_unit_tests = .false. ! Start by assuming all is well
  call test%set(verbose=verbose) ! Pass verbose mode to test

  ! Identity ANN for one input
  allocate( y(1) )
  call ANN_allocate(ANN, 2, [1,1])
  call set_layer(ANN, 1, reshape([1.],[1,1]), [0.], .false.)
  call ANN_apply([1.], y, ANN)
  call test%real_scalar(y(1), 1., 'Scalar identity')
  deallocate( y )
  call ANN_end(ANN)

  ! Summation ANN
  allocate( y(1) )
  call ANN_allocate(ANN, 2, [4,1])
  call set_layer(ANN, 1, reshape([1.,1.,1.,1.], [1,4]), [0.], .false.)
  call ANN_apply([-1.,0.,1.,2.], y, ANN)
  call test%real_scalar(y(1), 2., 'Summation')
  deallocate( y )
  call ANN_end(ANN)

  ! Identity ANN for vector input/output
  allocate( y(3) )
  call ANN_allocate(ANN, 2, [3,3])
  call set_layer(ANN, 1, reshape([1.,0.,0., &
                                  0.,1.,0., &
                                  0.,0.,1.], [3,3]), [0.,0.,0.], .false.)
  call ANN_apply([-1.,0.,1.], y, ANN)
  call test%real_arr(3, y, [-1.,0.,1.], 'Vector identity')
  deallocate( y )
  call ANN_end(ANN)

  ! Rectifying ANN for vector input/output
  allocate( y(3) )
  call ANN_allocate(ANN, 2, [3,3])
  call set_layer(ANN, 1, reshape([1.,0.,0., &
                                  0.,1.,0., &
                                  0.,0.,1.], [3,3]), [0.,0.,0.], .true.)
  call ANN_apply([-1.,0.,1.], y, ANN)
  call test%real_arr(3, y, [0.,0.,1.], 'Rectifier')
  deallocate( y )
  call ANN_end(ANN)

  ! The next 3 tests re-use the same network with 4 inputs, a 4-wide hidden layer, and one output
  allocate( y(1) )
  call ANN_allocate(ANN, 3, [4,4,1])

  ! 1 hidden layer: rectifier followed by summation
  ! Inputs: [-1,0,1,2]
  ! Rectified: [0,0,1,2]
  ! Sum: 3
  ! Outputs: 3
  call set_layer(ANN, 1, reshape([1.,0.,0.,0., &
                                  0.,1.,0.,0., &
                                  0.,0.,1.,0., &
                                  0.,0.,0.,1.], [4,4]), [0.,0.,0.,0.], .true.)
  call set_layer(ANN, 2, reshape([1.,1.,1.,1.], [1,4]), [0.], .false.)
  call ANN_apply([-1.,0.,1.,2.], y, ANN)
  call test%real_scalar(y(1), 3., 'Rectifier+summation')

  ! as above but with biases
  ! Inputs: [-2,-1,0,1]
  ! After bias: [-1,0,1,2] with b=1
  ! Rectified: [0,0,1,2]
  ! Sum: 3
  ! After bias: 6 with b=3
  ! Outputs: 6
  call set_layer(ANN, 1, reshape([1.,0.,0.,0., &
                                  0.,1.,0.,0., &
                                  0.,0.,1.,0., &
                                  0.,0.,0.,1.], [4,4]), [1.,1.,1.,1.], .true.)
  call set_layer(ANN, 2, reshape([1.,1.,1.,1.], [1,4]), [3.], .false.)
  call ANN_apply([-2.,-1.,0.,1.], y, ANN)
  call test%real_scalar(y(1), 6., 'Rectifier+summation+bias')

  ! as above but with normalization of inputs and outputs
  ! Inputs: [0,2,4,6]
  ! Normalized inputs: [-2,-1,0,1] (using mean=-4, norm=2)
  ! Normalized outputs: 6
  ! De-normalized output: 2 (using mean=-10, norm=2)
  call set_input_normalization(ANN, means=[4.,4.,4.,4.], norms=[0.5,0.5,0.5,0.5])
  call set_output_normalization(ANN, norms=[2.], means=[-10.])
  call ANN_apply([0.,2.,4.,6.], y, ANN)
  call test%real_scalar(y(1), 2., 'Rectifier+summation+bias+norms')

  deallocate( y )
  call ANN_end(ANN)

  ANN_unit_tests = test%summarize('ANN_unit_tests')

end function ANN_unit_tests

end module MOM_ANN
