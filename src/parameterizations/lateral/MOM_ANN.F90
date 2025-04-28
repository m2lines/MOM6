!> Implements the general purpose ANN.
module MOM_ANN

! This file is part of MOM6. See LICENSE.md for the license

use MOM_diag_mediator, only : diag_ctrl, time_type
use MOM_io, only : MOM_read_data
use MOM_error_handler, only : MOM_error, FATAL, MOM_mesg
!
implicit none ; private

#include <MOM_memory.h>

public ANN_init, ANN_apply, ANN_end

!> Type for a single Linear layer of ANN,
!! i.e. stores the matrix A and bias b
!! for matrix-vector multiplication
!! y = A*x + b.
type, private :: layer_type; private 
  integer :: output_width        !< Number of rows in matrix A
  integer :: input_width         !< Number of columns in matrix A
  logical :: activation = .True. !< If true, apply the default activation function

  real, allocatable :: A(:,:) !< Matrix in column-major order 
                              !! of size A(output_width, input_width)
  real, allocatable :: b(:)   !< bias vector of size output_width
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
    input_norms(:), & !< Array of length layer_sizes(1). By these values
                      !! each input feature will be divided before feeding into the ANN
    output_norms(:)   !< Array of length layer_sizes(num_layers). By these values
                      !! each output of the ANN will be multiplied
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

  integer :: i
  character(len=1) :: layer_num_str
  character(len=3) :: fieldname

  call MOM_mesg('ANN: init from ' // trim(NNfile), 2)

  ! Read the number of layers
  call MOM_read_data(NNfile, "num_layers", CS%num_layers)
  
  ! Read size of layers
  allocate(CS%layer_sizes(CS%num_layers))
  call MOM_read_data(NNfile, "layer_sizes", CS%layer_sizes)
  
  ! Read normalization factors
  allocate(CS%input_norms(CS%layer_sizes(1)))
  allocate(CS%output_norms(CS%layer_sizes(CS%num_layers)))

  call MOM_read_data(NNfile, 'input_norms', CS%input_norms)
  call MOM_read_data(NNfile, 'output_norms', CS%output_norms)
  
  ! Allocate the Linear transformations between layers.
  allocate(CS%layers(CS%num_layers-1))
  
  ! Allocate and read matrix A and bias b for each layer
  do i = 1,CS%num_layers-1
    CS%layers(i)%input_width = CS%layer_sizes(i)
    CS%layers(i)%output_width = CS%layer_sizes(i+1)

    allocate(CS%layers(i)%A(CS%layers(i)%output_width, CS%layers(i)%input_width), source=0.)
    ! Reading matrix A
    write(layer_num_str, '(I0)') i-1
    fieldname = trim('A') // trim(layer_num_str)
    call MOM_read_data(NNfile, fieldname, CS%layers(i)%A, &
                        (/1,1,1,1/),(/CS%layers(i)%output_width,CS%layers(i)%input_width,1,1/))

    allocate(CS%layers(i)%b(CS%layers(i)%output_width), source=0.)
    ! Reading bias b
    fieldname = trim('b') // trim(layer_num_str)
    call MOM_read_data(NNfile, fieldname, CS%layers(i)%b)
  enddo
  
  ! No activation function for the last layer
  CS%layers(CS%num_layers-1)%activation = .False.

  call ANN_test(CS, NNfile)

  call MOM_mesg('ANN: have been read from ' // trim(NNfile), 2)

end subroutine ANN_init

!> Test ANN by comparing the prediction with the test data.
subroutine ANN_test(CS, NNfile)
  type(ANN_CS), intent(inout)  :: CS     !< ANN control structure.
  character(*), intent(in)     :: NNfile !< The name of NetCDF file having neural network parameters

  real, dimension(:), allocatable :: x_test, y_test, y_pred
  real :: relative_error
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
  
  relative_error = maxval(abs(y_pred - y_test)) / maxval(abs(y_test))

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

  integer :: i

  deallocate(CS%layer_sizes)
  deallocate(CS%input_norms)
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
  type(ANN_CS), intent(in) :: CS !< ANN control structure

  real, dimension(CS%layer_sizes(1)), &
                  intent(in)  :: x !< input 
  real, dimension(CS%layer_sizes(CS%num_layers)), &
                  intent(out) :: y !< output 
  
  real, allocatable :: x_1(:), x_2(:) ! intermediate states. 
  integer :: i

  ! Normalize input
  allocate(x_1(CS%layer_sizes(1)))
  do i = 1,CS%layer_sizes(1)
      x_1(i) = x(i) / CS%input_norms(i)
  enddo
  
  ! Apply Linear layers
  do i = 1, CS%num_layers-1 
    allocate(x_2(CS%layer_sizes(i+1)))
    call Layer_apply(x_1, x_2, CS%layers(i))
    deallocate(x_1)
    allocate(x_1(CS%layer_sizes(i+1)))
    x_1 = x_2
    deallocate(x_2)
  enddo
  
  ! Un-normalize output
  do i = 1, CS%layer_sizes(CS%num_layers)
    y(i) = x_1(i) * CS%output_norms(i)
  enddo

  deallocate(x_1)
end subroutine ANN_apply

!> The default activation function
pure function activation_fn(x) result (y)
  real, intent(in)  :: x !< Scalar input value
  real :: y !< Scalar output value

  y = max(x, 0.0) ! ReLU activation
  
end function activation_fn

!> Applies linear layer to input data x and stores the result in y with 
!! y = A*x + b with optional application of the activation function.
subroutine Layer_apply(x, y, layer)
  type(layer_type), intent(in)  :: layer !< Linear layer
  real, dimension(layer%input_width), &
                    intent(in)  :: x     !< Input vector
  real, dimension(layer%output_width), &
                    intent(out) :: y     !< Output vector

  integer :: i, j

  do j=1,layer%output_width
    y(j) = 0.
    do i=1,layer%input_width
      ! Multiply by kernel
      y(j) = y(j) + ( x(i) * layer%A(j, i) )
    enddo
    ! Add bias
    y(j) = y(j) + layer%b(j)
    ! Apply activation function
    if (layer%activation) then
      y(j) = activation_fn(y(j))
    endif
  enddo
end subroutine Layer_apply

end module MOM_ANN