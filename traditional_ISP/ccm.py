import torch


def color_correction_matrix(frame: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Применяет Color Correction Matrix (CCM) к RGB изображению.

    Args:
        frame: RGB изображение (H x W x 3) после демозаика 
        config: словарь конфигурации, содержащий матрицу цветокоррекции

    Returns:
        torch.Tensor:
    """
    ccm = torch.tensor(config["ccm"]["ccm_matrix"], dtype=torch.float64, device=frame.device)
    H, W, _ = frame.shape
    
    # Преобразуем изображение в список RGB-векторов
    frame_flat = frame.reshape(-1, 3).double()

    # Применяем матрицу цветокоррекции
    frame_ccm = frame_flat @ ccm.T

    # Возвращаем исходную форму
    frame_ccm = frame_ccm.reshape(H, W, 3)

    return torch.clamp(frame_ccm, 0.0, 1.0)