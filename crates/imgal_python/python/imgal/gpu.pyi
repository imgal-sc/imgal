"""
GPU device setup and tensors.

This module provides the functions and structs needed to initialize a new
GPU device, create/read/write tensors, and convert them into GPU kernel
arguments.
"""

from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any, final

@final
class GPUTensor:
    """
    A tensor with a memory allocation on the GPU.

    The `GPUTensor` represents array data (*i.e.* a tensor) on an initialized
    GPU. An active `GPUTensor` will always have *some* GPU memory allocation
    and a shape.
    """
    @staticmethod
    def empty(shape: Sequence[int], dtype: str) -> GPUTensor:
        """
        Create a new empty GPU tensor.

        Creates a new empty GPU tensor with a size defined by `shape`.

        Args:
            The shape of the empty tensor.

        Returns:
            An initialized empty GPU tensor.
        """
    @staticmethod
    def empty_one(dtype: str) -> GPUTensor:
        """
        Create a new empty GPU tensor of size one.

        Creates a new empty GPU tensor of size one. A size one tensor is useful
        for functions that reduce an input tensor into one value.

        Returns:
            An initialized empty GPU tensor with a size of one.
        """
    @staticmethod
    def new(data: Any) -> GPUTensor:
        """
        Create a new GPU tensor with input data.

        Creates a new GPU tensor from the input *n*-dimensional image.

        Args:
            The input *n*-dimensional image.

        Returns:
            An initialized GPU tensor with the input data.
        """
    def read(self, /) -> Any:
        """
        Read data from the GPU tensor.

        Reads *n*-dimensional data from the tensor's reserved GPU memory.

        Returns:
            The tensor data.
        """
    @property
    def shape(self, /) -> list[int]:
        """
        Get the shape of the tensor.

        Gets the shape of the tensor. Once a tensor is initialized the shape is
        immutable.

        Returns:
            The shape of the tensor.
        """
    @property
    def size(self, /) -> int:
        """
        Get the size of the tensor.

        Gets the sieze of the tensor. Once a tensor is initialized the shape is
        immutable.

        Returns:
            The size of the tensor.
        """
    def write(self, /, data: Any) -> None:
        """
        Write data into the GPU tensor.

        Writes *n*-dimensional data into the tensor's reserved GPU memory.

        Args:
            The input *n*-dimensional image.
        """

def init_gpu() -> None:
    """
    Initialize the GPU and cache the compute client.

    Initializes the GPU and caches the compute client in a OnceLock. The client
    enables data transfer to and from the GPU.
    """

def __getattr__(name: str) -> Incomplete: ...
